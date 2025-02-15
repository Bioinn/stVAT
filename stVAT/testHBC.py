import pandas as pd
import anndata as ad
import numpy as np
import scipy 
import scanpy as sc
import torch
from stVAT import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
random.seed(42)

adata = sc.read_visium('/root/stVAT/data/Human_Breast_Cancer')
adata.var_names_make_unique()


sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)


train_adata = adata[:, adata.var["n_cells_by_counts"] > len(adata.obs.index) * 0.1]
train_counts = np.array(train_adata.X.todense())  
train_coords = train_adata.obs[['array_row', 'array_col']]


test_adata = sc.read_visium('/root/stVAT/data/Human_Breast_Cancer')
test_adata.var_names_make_unique()
integral_coords = test_adata.obs[['array_row', 'array_col']]

sc.pp.calculate_qc_metrics(test_adata, inplace=True)
sc.pp.filter_cells(test_adata, min_genes=200)
sc.pp.filter_genes(test_adata, min_cells=10)

test_counts = np.array(test_adata.X.todense())  
test_coords = test_adata.obs[['array_row', 'array_col']]


position_info = get_10X_position_info(integral_coords)


train_lr, train_hr, in_tissue_matrix = get10Xtrainset(train_counts, train_coords)
in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)

test_3D_data = get10Xtestset(test_counts, test_coords)


imputed_adata = stVAT(
    adata, test_3D_data, integral_coords, position_info, train_lr, train_hr, in_tissue_matrix,
    patch_size=8, num_heads=4, epoch=20, input_dim=1280
)


