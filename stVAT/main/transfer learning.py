import anndata as ad
from .network import stVAT as Model,VAE,VAE_Loss
from .func import *
import torch.nn as nn
def stVAT(adata, test_3D_data, integral_coords, position_info, train_lr, train_hr, in_tissue_matrix, test_adata,input_dim,
              patch_size=8, batch_size=512, vae_hidden_dim=32, vae_latent_dim=16, num_heads=4,
              epoch=500, lr=0.0001, vae_weight=0.005,k_size=5): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Model(patch_size=patch_size, embed_dim=patch_size*patch_size*4, num_heads=num_heads).to(device)
    vae = VAE(input_dim=input_dim, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim).to(device)
    vae_loss_fn = VAE_Loss()
    optimizer = torch.optim.AdamW(list(net.parameters()) + list(vae.parameters()), lr=lr, betas=(0.5, 0.6), eps=1e-6)  
    train_lr = data_pad(train_lr, patch_size)
    train_lr = torch.Tensor(train_lr.reshape((int(train_lr.shape[0]), 1, int(train_lr.shape[1]), int(train_lr.shape[2]))))
    train_hr = torch.Tensor(train_hr.reshape((int(train_hr.shape[0]), 1, int(train_hr.shape[1]), int(train_hr.shape[2]))))
    losses = []
    for epoch in range(epoch):
        loss_running = 0
        idx = 0
        for b_id, data in enumerate(data_iter(train_lr, train_hr, batch_size=batch_size), 0):
            idx += 1
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            # VAE 重构过程
            lr_flat = lr.view(lr.size(0), -1)  # 展平输入
            recon_lr, mu, logvar = vae(lr_flat)
            recon_lr = recon_lr.view_as(lr)
            vae_loss = vae_loss_fn(recon_lr, lr, mu, logvar)
            combined_image = 0.3 * recon_lr + (1 - 0.3) * lr
            pre_hr = net(combined_image)
            task_loss = criterion(pre_hr, hr, in_tissue_matrix)
            loss = task_loss + vae_weight * vae_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_running += loss.item()
        avg_loss = loss_running / idx
        losses.append(avg_loss)
    b, h, w = test_3D_data.shape
    test_3D_data = data_pad(test_3D_data, patch_size=patch_size)
    test_3D_data = torch.Tensor(test_3D_data.reshape((b, 1, test_3D_data.shape[1], test_3D_data.shape[2])))
    pre_3D_data = []
    for i in range(0, test_3D_data.shape[0], 128):
        with torch.no_grad():
            data = test_3D_data[i:min((i + 128), test_3D_data.shape[0]), :, :, :]
            data = data.to(device)
            pre_data = net(data)
            pre_data = get_test_data(pre_data, is_pad=True, train_lr_h=h, train_lr_w=w)  # （b, 2h, 2w）
            pre_3D_data.append(pre_data)
    pre_3D_data = torch.cat(pre_3D_data, dim=0)
    imputed_counts, imputed_coords = img2expr(pre_3D_data, test_adata.var_names, integral_coords, position_info)
    imputed_adata = ad.AnnData(X=imputed_counts, obs=imputed_coords)
    adata.X[adata.X < 0.5] = 0
    return imputed_adata
