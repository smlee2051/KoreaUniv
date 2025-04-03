import os
import sys
import time
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import openai  # 

from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<Model Name>\"")
        return
    selected_model = " ".join(sys.argv[1:])  # ex) "LLMTime GPT-3.5"

    openai.api_key = 'OPENAI_API_KEY'
    openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '../data/server_data.csv')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    os.environ['OMP_NUM_THREADS'] = '4'
    torch.cuda.empty_cache()

    gpt3_hypers = dict(
        temp=0.7,
        alpha=0.95,
        beta=0.3,
        basic=False,
        settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True, minus_sign='-', missing_str=None)
    )

    gpt4_hypers = dict(
        alpha=0.3,
        basic=True,
        temp=1.0,
        top_p=0.8,
        settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
    )

    promptcast_hypers = dict(
        temp=0.7,
        settings=SerializerSettings(base=10, prec=0, signed=True, time_sep=', ', bit_sep='',
                                    plus_sign='', minus_sign='-', half_bin_correction=False, decimal_point='')
    )

    model_hypers = {
        'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
        'LLMTime GPT-4o': {'model': 'gpt-4o', **gpt4_hypers},
        'PromptCast GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **promptcast_hypers},
    }

    model_predict_fns = {
        'LLMTime GPT-3.5': get_llmtime_predictions_data,
        'LLMTime GPT-4o': get_llmtime_predictions_data,
        'PromptCast GPT-3.5': get_promptcast_predictions_data,
    }

    if selected_model not in model_hypers or selected_model not in model_predict_fns:
        print(f"Error: Model '{selected_model}' not supported.")
        return

    # Data
    data_series = pd.read_csv(data_path)
    person_ids = data_series['person_id'].unique()

    results = {}
    metrics = []
    start_time = time.time()

    for person_id in person_ids:
        print(f"Processing person_id: {person_id}")
        person_data = data_series.query(f'person_id == {person_id}').set_index('date')['tar']
        test_size = 7
        train = person_data[:-test_size]
        test = person_data[-test_size:]

        # update
        model_hyper = model_hypers[selected_model]
        model_hyper.update({'dataset_name': f'CustomDataset_person_{person_id}'})
        hypers = list(grid_iter(model_hyper))
        num_samples = 10

        pred_fn = model_predict_fns[selected_model]
        pred_dict = get_autotuned_predictions_data(
            train, test, hypers, num_samples, pred_fn, verbose=False, parallel=False
        )

        results[(person_id, selected_model)] = pred_dict

        # metric
        pred = pred_dict['median']
        mse = mean_squared_error(test, pred)
        mae = mean_absolute_error(test, pred)

        metrics.append({
            'person_id': person_id,
            'model': selected_model,
            'mse': mse,
            'mae': mae
        })

    # save
    pd.DataFrame(metrics).to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    with open(os.path.join(results_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"Processing complete. Elapsed time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
