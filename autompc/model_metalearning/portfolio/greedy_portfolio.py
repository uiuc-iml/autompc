import os
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from autompc.model_metalearning.meta_utils import load_data, load_cfg, load_matrix
from autompc.model_metalearning.meta_utils import meta_data
import autompc.model_metalearning.portfolio_util

data_path = '/home/baoyul2/scratch/autompc/autompc/model_metalearning/meta_data'
cfg_path = '/home/baoyul2/scratch/autompc/autompc/model_metalearning/meta_cfg'
matrix_path = '/home/baoyul2/scratch/autompc/autompc/model_metalearning/meta_matrix'
portfolio_path = '/home/baoyul2/scratch/autompc/autompc/model_metalearning/meta_portfolio'

def calculate_generalization_error(matrix, P):
    # The extracted matrix based on P
    candidate_matrix = matrix[P]
    # min generalization errors for each dataset
    minima = candidate_matrix.min(axis=1)
    # generalization error of P across all meta-datasets
    generalization_error = minima.sum()
    return generalization_error

def build_portfolio(
    matrix: pd.DataFrame,
    portfolio_size: int,
    meta_names: list
):
    portfolio = []
    configurations = []
    for cfg_name in meta_names:
        cfg = load_cfg(cfg_path, cfg_name)
        configurations.append(cfg)
    
    cfg_index = list(range(len(configurations)))
    portfolio_index = [] # Store the index of corresponding candidate
    # Construct Portfolio
    while len(portfolio) < portfolio_size:
        errors = [] # Store the generalization error for each P across all meta-datasets
        for i in range(len(configurations)):
            P = portfolio_index.copy()
            P.append(i)
            generalization_error = calculate_generalization_error(matrix, P)
            errors.append(generalization_error)
        
        # Add the candidate with the min error to portfolio
        min_index = errors.index(min(errors))
        cfg_min_index = cfg_index[min_index]
        portfolio.append(configurations[min_index])
        portfolio_index.append(cfg_min_index)
        print(meta_names[cfg_min_index])
        
        # Remove the chosen candidate from the configurations
        configurations.pop(min_index)
        cfg_index.pop(min_index)
    
    # Save portfolio
    output_file_name = os.path.join(portfolio_path, 'portfolio_' + str(portfolio_size) + '.pkl')
    print("Dumping to ", output_file_name)
    with open(output_file_name, 'wb') as fh:
        pickle.dump(portfolio, fh)
    
    return portfolio
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--portfolio_size", type=int)
    args = parser.parse_args()
    
    meta_names = meta_data
    matrix = load_matrix(matrix_path)
    portfolio = build_portfolio(matrix=matrix, portfolio_size=args.portfolio_size, meta_names=meta_names)
    print(portfolio)
    print('\n')
    