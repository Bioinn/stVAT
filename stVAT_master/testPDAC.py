from stVAT import *
import pandas as pd
import anndata as ad
import numpy as np
import scipy 
import scanpy as sc
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

counts = pd.read_csv('/root/stVAT/PDAC/A1/PDAC_A_ST1.csv',index_col=0)  
coords = pd.read_csv('/root/stVAT/PDAC/A1/spatial/coords.csv', index_col=0)
coords['array_row'] = coords['array_row'].astype(int)
coords['array_col'] = coords['array_col'].astype(int)
adata = ad.AnnData(X=counts.values, obs=coords, var=pd.DataFrame(index=counts.columns.values))

integral_coords = adata.obs[['array_row','array_col']]
adata.var_names_make_unique()

integral_coords = adata.obs[['array_row', 'array_col']]
adata.var_names_make_unique()
position_info = get_ST_position_info(integral_coords)

sc.pp.calculate_qc_metrics(adata, inplace=True)

sc.pp.filter_cells(adata, min_genes=10)  
sc.pp.filter_genes(adata, min_cells=10)  

train_adata = adata[:, adata.var["n_cells_by_counts"] > len(adata.obs.index) * 0.1]

train_counts = np.array(train_adata.X)  # ndarray对象
train_coords = train_adata.obs[['array_row', 'array_col']] 

train_lr, train_hr, in_tissue_matrix = get_train_data(train_counts, train_coords)
in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)


test_counts = np.array(adata.X)
test_coords = adata.obs[['array_row', 'array_col']]
test_3D_data = getSTtestset(test_counts, test_coords)

imputed_adata = stVAT(adata, test_3D_data, integral_coords, position_info, train_lr, train_hr, in_tissue_matrix, patch_size=8, num_heads=4,epoch=20,input_dim=256)
