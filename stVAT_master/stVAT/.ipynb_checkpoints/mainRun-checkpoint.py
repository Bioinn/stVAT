import anndata as ad
from .network import stVAT_LatentFusion as Model
from .func import *
import torch
import torch.nn.functional as F


def _pad_or_crop_3d_to_shape(x, target_h, target_w):

    x = torch.Tensor(x)

    _, h, w = x.shape

    # crop
    if h > target_h:
        x = x[:, :target_h, :]
    if w > target_w:
        x = x[:, :, :target_w]

    _, h, w = x.shape

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    # F.pad 的顺序是 (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h))

    return x.detach().numpy()


def stVAT(
    adata,
    test_3D_data,
    integral_coords,
    position_info,
    train_lr,
    train_hr,
    in_tissue_matrix,
    input_dim=None,
    patch_size=8,
    batch_size=512,
    vae_hidden_dim=32,
    vae_latent_dim=16,
    num_heads=4,
    epoch=500,
    lr=0.0001,
    vae_weight=0.005,
    k_size=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ====================================================
    # 1. 先根据 test_3D_data 确定推理时的 padding 尺寸
    # ====================================================
    b_test, test_h_raw, test_w_raw = test_3D_data.shape

    test_3D_pad_np = data_pad(
        test_3D_data,
        patch_size=patch_size
    )

    target_h = int(test_3D_pad_np.shape[1])
    target_w = int(test_3D_pad_np.shape[2])

    print(f"[stVAT] raw test size: ({test_h_raw}, {test_w_raw})")
    print(f"[stVAT] padded test size used for VAE/ViT: ({target_h}, {target_w})")

    # ====================================================
    # 2. train_lr 也 pad 到和 test_3D_data 一样的尺寸
    # ====================================================
    train_lr = data_pad(
        train_lr,
        patch_size=patch_size
    )

    train_lr = _pad_or_crop_3d_to_shape(
        train_lr,
        target_h,
        target_w
    )

    if input_dim is None:
        input_dim = int(target_h * target_w)

    print(f"[stVAT] VAE input_dim: {input_dim}")
    print(f"[stVAT] train_lr after pad/crop: {train_lr.shape}")
    print(f"[stVAT] train_hr shape: {train_hr.shape}")

    train_lr = torch.Tensor(
        train_lr.reshape(
            int(train_lr.shape[0]),
            1,
            int(train_lr.shape[1]),
            int(train_lr.shape[2])
        )
    )

    train_hr = torch.Tensor(
        train_hr.reshape(
            int(train_hr.shape[0]),
            1,
            int(train_hr.shape[1]),
            int(train_hr.shape[2])
        )
    )

    # ====================================================
    # 3. 构建模型
    # ====================================================
    net = Model(
        patch_size=patch_size,
        embed_dim=patch_size * patch_size * 4,
        num_heads=num_heads,
        input_dim=input_dim,
        vae_hidden_dim=vae_hidden_dim,
        vae_latent_dim=vae_latent_dim,
        k_size=k_size
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        betas=(0.5, 0.6),
        eps=1e-6
    )

    if not torch.is_tensor(in_tissue_matrix):
        in_tissue_matrix = torch.Tensor(in_tissue_matrix)

    in_tissue_matrix = in_tissue_matrix.to(device)

    # ====================================================
    # 4. 训练
    # ====================================================
    losses = []

    for ep in range(epoch):
        loss_running = 0
        idx = 0

        for b_id, data in enumerate(
            data_iter(train_lr, train_hr, batch_size=batch_size),
            0
        ):
            idx += 1

            lr_batch, hr_batch = data

            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            pre_hr, vae_loss = net(
                lr_batch,
                return_vae_loss=True
            )

            task_loss = criterion(
                pre_hr,
                hr_batch,
                in_tissue_matrix
            )

            loss = task_loss + vae_weight * vae_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_running += loss.item()

        avg_loss = loss_running / idx
        losses.append(avg_loss)

        print(f'epoch:{ep + 1}, loss:{round(avg_loss, 3)}')
    torch.save(net.state_dict(), 'stVAT_last_epoch.params')

    # ====================================================
    # 5. 推理
    # ====================================================
    test_3D_data = torch.Tensor(
        test_3D_pad_np.reshape(
            b_test,
            1,
            target_h,
            target_w
        )
    )

    pre_3D_data = []

    net.eval()

    for i in range(0, test_3D_data.shape[0], 128):
        with torch.no_grad():
            data = test_3D_data[
                i:min((i + 128), test_3D_data.shape[0]),
                :, :, :
            ].to(device)

            pre_data = net(data)

            if isinstance(pre_data, tuple):
                pre_data = pre_data[0]

            pre_data = get_test_data(
                pre_data,
                is_pad=True,
                train_lr_h=test_h_raw,
                train_lr_w=test_w_raw
            )

            pre_3D_data.append(pre_data.cpu())

    pre_3D_data = torch.cat(pre_3D_data, dim=0)

    # ====================================================
    # 6. 转成表达矩阵
    # ====================================================
    imputed_counts, imputed_coords = img2expr(
        pre_3D_data,
        adata.var_names,
        integral_coords,
        position_info
    )

    imputed_adata = ad.AnnData(
        X=imputed_counts,
        obs=imputed_coords
    )

    return imputed_adata
