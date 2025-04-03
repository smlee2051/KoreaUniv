import os
import sys
import torch
import random
import numpy as np
import itertools
from exp.exp_main import Exp_Main


def log_to_txt(file_path, text):
    with open(file_path, mode='a') as file:
        file.write(text + '\n')


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <ModelName>")
        return
    model_name = sys.argv[1]

    # base_dir 
    base_dir = os.path.dirname(os.path.abspath(__file__))

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    class Args:
        is_training = 1
        model_id = 'test'
        model = model_name
        data = 'custom'
        root_path = os.path.join(base_dir, 'data')
        data_path = 'server_data.csv'
        features = 'S'
        target = 'tar'
        freq = 'd'
        checkpoints = os.path.join(base_dir, 'checkpoints')
        seq_len = 14
        label_len = seq_len
        pred_len = 7
        bucket_size = 4
        n_hashes = 4
        enc_in = 1
        dec_in = 1
        c_out = 1
        d_model = 512
        n_heads = 8
        e_layers = 2
        d_layers = 1
        d_ff = 2048
        moving_avg = 25
        factor = 1
        distil = True
        dropout = 0.05
        embed = 'timeF'
        activation = 'gelu'
        output_attention = False
        do_predict = True
        num_workers = 10
        itr = 2
        train_epochs = 100
        batch_size = 32
        patience = 3
        learning_rate = 0.0001
        des = 'test'
        loss = 'mse'
        lradj = 'type1'
        use_amp = False
        use_gpu = True
        gpu = 0
        use_multi_gpu = False
        devices = '0,1,2,3'
        search_type = 'random'

    args = Args()

    # GPU
    if args.use_gpu and torch.cuda.is_available():
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids if int(id_) < torch.cuda.device_count()]
            if not args.device_ids:
                raise ValueError("All specified GPU IDs are invalid or not available")
            args.gpu = args.device_ids[0]
        else:
            if args.gpu >= torch.cuda.device_count():
                raise ValueError(f"Specified GPU ID {args.gpu} is not available.")
            args.device_ids = [args.gpu]
    else:
        args.use_gpu = False
        args.device_ids = []
        print("Warning: No GPU available or GPU usage is disabled. Running on CPU.")

    param_grid = {
        'batch_size': [8, 32],
        'seq_len': [14, 21, 28],
        'd_model': [128, 256, 1024],
        'n_heads': [8, 16],
        'e_layers': [2, 4, 6],
        'd_layers': [1, 2]
    }

    if args.search_type == 'grid':
        param_combinations = list(itertools.product(*param_grid.values()))
    elif args.search_type == 'random':
        param_combinations = list(itertools.product(*param_grid.values()))
        random.shuffle(param_combinations)
        total_combinations = len(param_combinations)
        random_search_trials = int(total_combinations * 0.2)
        param_combinations = param_combinations[:random_search_trials]

    best_val_loss = float('inf')
    best_params = None

    result_dir = os.path.join(base_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    Exp = Exp_Main

    if args.is_training:
        for param_combo in param_combinations:
            for i, key in enumerate(param_grid.keys()):
                setattr(args, key, param_combo[i])
            args.label_len = args.seq_len
            setting = f"{args.model_id}_{args.model}"
            exp = Exp(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            val_loss = float(exp.train(setting))

            txt_file_path = os.path.join(result_dir, f'val_losses_{setting}.txt')
            param_combo_dict = {key: value for key, value in zip(param_grid.keys(), param_combo)}
            text = f"Params: {param_combo_dict}, Validation Loss: {val_loss:.6f}"
            log_to_txt(txt_file_path, text)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = param_combo

        print(f'Best hyperparameters: {best_params}')
        print(f'Best validation loss: {best_val_loss:.6f}')

        for i, key in enumerate(param_grid.keys()):
            setattr(args, key, best_params[i])
        args.label_len = args.seq_len
        setting = f"{args.model_id}_{args.model}"
        exp = Exp(args)
        exp.train(setting)
        exp.test(setting)


if __name__ == '__main__':
    main()