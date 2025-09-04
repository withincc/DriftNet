import numpy as np
import torch
from torch import nn,Tensor
from functools import partial
import random
'''
Datetime:2025
Author:Yongxiang Chen
'''
seed=2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):

    def __init__(self, OceanData_size, DriftData_dim,OceanData_dim,  hidden_dim,patch_size):
        super().__init__()
        OceanData_size = (OceanData_size, OceanData_size)
        patch_size = (patch_size, patch_size)
        self.img_size = OceanData_size
        self.patch_size = patch_size
        self.grid_size = (OceanData_size[0] // patch_size[0], OceanData_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj_OceanData = nn.Conv2d(OceanData_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_Drift=nn.Linear(DriftData_dim,hidden_dim)
    def forward(self, DriftData,OceanData):
        DriftData = self.proj_Drift(DriftData)
        OceanData = self.proj_OceanData(OceanData).flatten(2).transpose(1, 2)
        return DriftData,OceanData
class MultiHead_TADA(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 lambda_init=0.1):
        super(MultiHead_TADA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.scale = qk_scale or head_dim ** -0.5
        self.W_q = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.W_k = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_o = nn.Linear(dim, dim, bias=qkv_bias)
        # lamda shapes, num_heads, head_dim, so that a lambda can be assigned to each head
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, head_dim))
        self.lambda_init = lambda_init
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        # self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        # nn.init.constant_(self.rms_scale, 1.0)
    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]

    def forward(self, DriftData,OceanData)->Tensor:
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)
        lambda_val = lambda_val.unsqueeze(-1).unsqueeze(-1)
        Q=self.W_q(DriftData)
        K=self.W_k(OceanData)
        V=self.W_v(OceanData)
        # 划分Head
        batch, DriftData_N, dim = DriftData.shape
        _, OceanData_N, _ = OceanData.shape
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, -1).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, -1).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, -1).transpose(1, 2)

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)
        A1=(Q1 @ K1.transpose(-1, -2)) * self.scale
        A2=(Q2 @ K2.transpose(-1, -2)) * self.scale

        A1_softmax = A1.softmax( dim=-1)
        A2_softmax = A2.softmax( dim=-1)
        att=(A1_softmax - lambda_val * A2_softmax)
        Out =att  @ V
        O_concat = Out.transpose(1, 2).contiguous().view(batch, DriftData_N, -1)
        result = self.W_o(O_concat)

        return result

class Mlp(nn.Module):


    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHead_TADA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, DriftData,OceanData):
        DriftData = DriftData + self.drop_path(self.attn(self.norm1(DriftData),OceanData))
        DriftData = DriftData + self.drop_path(self.mlp(self.norm1(DriftData)))
        return DriftData,OceanData
class DriftNet(nn.Module):
    def __init__(self, OceanData_size=21,
                        DriftData_dim=9,
                        OceanData_dim=9,
                        patch_size=1,
                        hidden_dim=48,
                        depth=24,
                        num_heads=1,
                        output_dim=2,
                 mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.1,
                 attn_drop_ratio=0.1, drop_path_ratio=0.1, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(DriftNet, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drift_embedding = embed_layer(OceanData_size=OceanData_size, patch_size=patch_size, DriftData_dim=DriftData_dim,OceanData_dim=OceanData_dim, hidden_dim=hidden_dim)
        num_patches = self.drift_embedding.num_patches
        self.Drift_token = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.output_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)
    def forward_features(self, DriftData,OceanData):
        DriftData=DriftData.unsqueeze(1)
        DriftData,OceanData = self.drift_embedding(DriftData,OceanData)  # [B, patch_num, embed_dim]
        DriftData=self.Drift_token*DriftData
        OceanData = self.pos_drop(OceanData + self.pos_embed)
        for block in self.blocks:
            DriftData,OceanData = block(DriftData,OceanData)
        return DriftData.squeeze(1)
    def forward(self, DriftData,OceanData):
        Drift_features = self.forward_features(DriftData,OceanData)
        x = self.head(Drift_features)
        return x
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def driftnet(hidden_dim=48,depth=24,num_heads=8):
    model = DriftNet(OceanData_size=21,
                        DriftData_dim=9,
                        OceanData_dim=9,
                        patch_size=1,
                        hidden_dim=hidden_dim,
                        depth=depth,
                        num_heads=num_heads,
                        output_dim=2)
    return model
if __name__ == '__main__':
    model = driftnet()
    print(model)