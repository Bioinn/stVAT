import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# VAE Implementation
class VAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.bn3(self.fc3(z)))
        return self.fc4(h3)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x.view_as(x), mu, logvar  

class VAE_Loss(nn.Module):
    def __init__(self,max_kl_weight=1.0, scale_factor=0.99):
        super(VAE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.kl_weight = 1e-4  
        self.max_kl_weight = max_kl_weight
        self.scale_factor = scale_factor

    def forward(self, recon_x, x, mu, logvar):
        BCE = self.mse_loss(recon_x, x.view_as(recon_x))  
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl_weight = min(self.max_kl_weight, self.kl_weight / self.scale_factor)
        return BCE + self.kl_weight * KLD

# VisionTransformer Implementation
class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=16, in_channel=1, embed_dim=1024):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # conv: (b, 1, h, w) -> (b, 1024, h/16, w/16)
        # flatten: (b, 256, h/16, w/16) -> (b, 256, hw/256)
        # reshape:(b, 256, hw/256) -> (b, hw/256, 256) (batch, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class ECALayer(torch.nn.Module):
    def __init__(self, channels, k_size=5):
        super(ECALayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [B, N, E]
        b, n, c = x.size()
        # Apply adaptive average pooling to reduce dimensions
        y = self.avg_pool(x.transpose(1, 2))  # Shape [B, C, N]
        y = self.conv(y)  # Shape [B, C, N]
        y = self.sigmoid(y)  # Shape [B, C, N]
        return x * y.transpose(1, 2)  # Shape [B, N, C]

class Attention(torch.nn.Module):
    #dim = patch_size * patch_size * 4, which is quadrupled in PatchEmbed.
    def __init__(self, dim=16, num_heads=8, drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim*3, bias=True)
        self.drop1 = torch.nn.Dropout(drop_ratio)
        self.proj = torch.nn.Linear(dim, dim)
        self.drop2 = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(D / self.num_heads)).permute(2, 0, 3, 1, 4)
        # (batch, num_heads, num_patches, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, num_patches, num_patches)
        att = att.softmax(dim=-1)
        att = self.drop1(att)
        x = (att @ v).transpose(1, 2).flatten(2)  # B,N,dim
        x = self.drop2(self.proj(x))
        return x
class Mlp(torch.nn.Module):
    def __init__(self, in_dim=1024, drop_ratio=0.):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, in_dim*2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(in_dim*2, in_dim)
        self.drop = torch.nn.Dropout(drop_ratio)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):
    def __init__(self, in_dim=1024, num_heads=8, drop_ratio=0., k_size=5):
        super(Block, self).__init__()
        # This step is very important, otherwise it will be difficult to converge.
        self.norm1 = nn.LayerNorm(in_dim)
        self.eca1 = ECALayer(channels=in_dim, k_size=k_size)
        self.attn = Attention(dim=in_dim, num_heads=num_heads, drop_ratio=drop_ratio)
        self.eca2 = ECALayer(channels=in_dim, k_size=k_size)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = Mlp(in_dim=in_dim, drop_ratio=drop_ratio)
        self.eca3 = ECALayer(channels=in_dim, k_size=k_size)
        self.drop = nn.Dropout(0.)

    def forward(self, x):
        x = self.eca1(self.norm1(x))  
        x = x + self.drop(self.attn(x))
        x = self.eca2(x)  
        x = self.norm2(x)
        x = x + self.drop(self.mlp(x))
        x = self.eca3(x)  
        return x
def absolute_position_encoding(seq_len, embed_dim):
    """
    Generate absolute position coding
    :param seq_len: Sequence length
    :param embed_dim: PatchEmbed length
    :return: absolute position coding
    """
    # (10000 ** ((2 * i) / embed_dim))
    seq_len = int(seq_len)
    pos_enc = torch.zeros((seq_len, embed_dim))
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (2*i / embed_dim)))
            if i + 1 < embed_dim:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (2*i / embed_dim)))
    return pos_enc
class stVAT(torch.nn.Module):
    def __init__(self,vae_weight=0.01, patch_size=16, in_c=1, embed_dim=1024, depth=12, num_heads=8,input_dim=256, drop_ratio=0.,vae_hidden_dim=64, vae_latent_dim=16,k_size=5):
        super(stVAT, self).__init__()
        self.vae = VAE(input_dim=input_dim, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim)
        self.vae_weight = vae_weight
        self.vae_loss = VAE_Loss()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.eca_layer = ECALayer(channels=embed_dim)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channel=in_c, embed_dim=embed_dim)
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)
        # depth transformer code blocks.
        self.blocks = torch.nn.Sequential(*[
            Block(in_dim=embed_dim, num_heads=num_heads, drop_ratio=drop_ratio,k_size=k_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b, _, h, w = x.shape
        x_flat = x.view(b, -1)
        input_dim = x_flat.shape[1]
        if input_dim != self.vae.fc1.in_features:
            self.vae = VAE(input_dim=input_dim).to(device)
        recon_x, mu, logvar = self.vae(x)
        vae_loss = self.vae_loss(recon_x, x, mu, logvar)

        num_patches = (h // self.patch_size) * (w // self.patch_size)
        pos = absolute_position_encoding(num_patches, self.embed_dim).to(device)
        x = self.patch_embed(x)
        x = self.eca_layer(x)
        x = self.pos_drop(x + pos)
        x = self.blocks(x)
        x = x.reshape(b, -1, int(self.embed_dim // 4), 4).transpose(1, 3).reshape(b, 4, int(self.embed_dim // 4),
                                                                                  int(h / self.patch_size),
                                                                                  int(w / self.patch_size))
        fina_x = torch.zeros((b, 4, h, w)).to(device)
        k = 0
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                fina_x[:, :, i::self.patch_size, j::self.patch_size] = x[:, :, k, :, :]
                k += 1
        return fina_x
        b, _, h, w = x.shape  # Compute height and width here