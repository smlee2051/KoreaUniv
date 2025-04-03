import os
import sys
import time
import torch
import random
import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, load_content

LLM_DIM_MAP = {
    'LLAMA': 4096,
    'GPT2': 768,
    'BERT': 768
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <LLM_MODEL>  # Options: LLAMA, GPT2, BERT")
        return

    llm_model = sys.argv[1].upper()
    if llm_model not in LLM_DIM_MAP:
        print(f"Unsupported LLM model '{llm_model}'. Choose from: {list(LLM_DIM_MAP.keys())}")
        return
    llm_dim = LLM_DIM_MAP[llm_model]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(base_dir, '../data')

    class Args:
        task_name = 'short_term_forecast'
        is_training = 1
        model_id = 'test'
        model_comment = 'none'
        model = 'TimeLLM'
        seed = 2021
        data = 'custom'
        root_path = root_path
        data_path = 'server_data.csv'
        features = 'S'
        target = 'tar'
        loader = 'modal'
        freq = 'd'
        checkpoints = os.path.join(base_dir, 'checkpoints')
        seq_len = 14
        label_len = 14
        pred_len = 7
        seasonal_patterns = 'Monthly'
        enc_in = 1
        dec_in = 1
        c_out = 1
        d_model = 16
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 32
        moving_avg = 25
        factor = 1
        dropout = 0.1
        embed = 'timeF'
        activation = 'gelu'
        output_attention = False
        patch_len = 16
        stride = 8
        prompt_domain = 0
        llm_model = llm_model
        llm_dim = llm_dim
        num_workers = 10
        itr = 1
        train_epochs = 10
        align_epochs = 10
        batch_size = 32
        eval_batch_size = 8
        patience = 5
        learning_rate = 0.0001
        des = 'test'
        loss = 'MSE'
        lradj = 'type1'
        pct_start = 0.2
        use_amp = False
        llm_layers = 6
        percent = 100
        content = None

    args = Args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=os.path.join(base_dir, 'ds_config_zero2.json'))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    for ii in range(args.itr):
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_{args.des}_{ii}"

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        model = TimeLLM.Model(args).float()

        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )

        for epoch in range(args.train_epochs):
            train_loss = []
            model.train()
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                accelerator.backward(loss)
                model_optim.step()
                scheduler.step()
                train_loss.append(loss.item())

            vali_loss, _ = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.6f} | Vali Loss: {vali_loss:.6f} | Test Loss: {test_loss:.6f} | MAE: {test_mae_loss:.6f}")

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

        all_test_predictions = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                all_test_predictions.append(outputs.detach().cpu().numpy())

        all_test_predictions = np.concatenate(all_test_predictions, axis=0)
        output_path = os.path.join(base_dir, f'final_test_predictions_{llm_model}.npy')
        np.save(output_path, all_test_predictions)
        accelerator.print(f"Final predictions saved to {output_path}")

if __name__ == '__main__':
    main()