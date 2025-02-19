## Adding property-conditioning of https://arxiv.org/pdf/2210.12765

import os
import argparse
from pathlib import Path
import numpy as np
from props.properties import penalized_logp, MolLogP_smiles, qed_smiles, ExactMolWt_smiles, compute_flat_properties, compute_flat_properties_nogap
from model.mxmnet import MXMNet
from data.dataset import get_cond_datasets
from evaluate.MCD.evaluator import TaskModel
DATA_DIR = "../resource/data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='zinc')
    parser.add_argument("--MAX_LEN", type=int, default=250)
    params = parser.parse_args()

    raw_dir = f"{DATA_DIR}/{params.dataset_name}"
    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=False)

    properties_path = os.path.join(raw_dir, f"properties_gflownet_nogap_train.npy")
    if not os.path.exists(properties_path):
        for split, dataset in zip(['train', 'valid', 'test'], datasets):
            # No Gap
            properties = compute_flat_properties_nogap(dataset.smiles_list, device='cuda')

            properties_path = os.path.join(raw_dir, f"properties_gflownet_nogap_{split}.npy")
            if os.path.exists(properties_path):
                os.remove(properties_path)
            with open(properties_path, 'wb') as f:
                np.save(f, properties)
            print('saved no-gap')

            # With Gap
            is_valid, properties = compute_flat_properties(dataset.smiles_list, device='cuda')

            properties_path = os.path.join(raw_dir, f"properties_gflownet_{split}.npy")
            if os.path.exists(properties_path):
                os.remove(properties_path)
            with open(properties_path, 'wb') as f:
                np.save(f, properties)

            is_valid_path = os.path.join(raw_dir, f"is_valid_gflownet_{split}.npy")
            if os.path.exists(is_valid_path):
                os.remove(is_valid_path)
            with open(is_valid_path, 'wb') as f:
                np.save(f, is_valid)
            print('saved with-gap')

    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=True, gflownet_realgap=False)

    datasets = get_cond_datasets(dataset_name=params.dataset_name, raw_dir=raw_dir, randomize_order=False, 
                MAX_LEN=params.MAX_LEN, scaling_type='std', gflownet=True, gflownet_realgap=True)

    print("Random forest predictors")
    for i in range(datasets[0].properties.shape[1]):
        model_path = f"../resource/data/{params.dataset_name}/gflownet_forest_model_{i}.csv.gz"
        forest_model = TaskModel(model_path, datasets[0].smiles_list, datasets[0].properties, 
            smiles_list_valid=datasets[2].smiles_list, properties_valid=datasets[2].properties, 
            i = i, task_type = 'regression')