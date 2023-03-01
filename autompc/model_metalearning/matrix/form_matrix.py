import pickle
import os
import pandas as pd

from autompc.model_metalearning.meta_utils import load_sub_matrix, normalize_matrix

matrix_path = '/home/baoyul2/scratch/autompc/autompc/model_metalearning/meta_matrix'

matrix_1 = load_sub_matrix(matrix_path, 'matrix_1')
matrix_2 = load_sub_matrix(matrix_path, 'matrix_2')
matrix_3 = load_sub_matrix(matrix_path, 'matrix_3')
matrix_4 = load_sub_matrix(matrix_path, 'matrix_4')

#print(matrix_1)
#print(matrix_2)
#print(matrix_3)
#print(matrix_4)

frame = [matrix_1, matrix_2, matrix_3, matrix_4]
matrix = pd.concat(frame)
# Min-Max normalization for each row
matrix = normalize_matrix(matrix)

# Save matrix
output_file_name = os.path.join(matrix_path, 'matrix.pkl')
print("Dumping to ", output_file_name)
with open(output_file_name, 'wb') as fh:
    pickle.dump(matrix, fh)
print(matrix.to_string())
