"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
import sys
import os
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from SoftMoE import SoftMoE
from Codes import *
import numpy as np

###################################################
###################################################

def diff_syndrome(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    # x_bin = sign_to_bin(x) if -1 in x else x
    x_bin = x
    
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    return tmp

def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

###################################################
###################################################
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class PlusConv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
            super().__init__(in_channels, out_channels, kernel_size=3, padding=0, **kwargs)
            mask = torch.tensor([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=torch.float32)
            self.register_buffer('mask', mask.view(1, 1, 3, 3).expand_as(self.weight).clone())

        def forward(self, x):
            masked_weight = self.weight * self.mask
            x = F.pad(x, (1, 1, 1, 1), mode='circular')
            return F.conv2d(x, masked_weight, self.bias, self.stride, 
                            self.padding, self.dilation, self.groups)
    
class ToricViTEmbedding(nn.Module):
    def __init__(self, L, d_model):
        super().__init__()
        self.L = L
        self.grid_dim = 2 * L
        self.conv_plus = PlusConv2d(1, d_model)
        
        self.num_qubits = 2 * L * L
        # [FIXED HERE] Tạo index mask
        grid_indices = torch.arange(self.grid_dim * self.grid_dim).view(self.grid_dim, self.grid_dim)
        
        # rows, cols ban đầu shape (2L, 1) và (1, 2L)
        rows_raw = torch.arange(self.grid_dim).view(-1, 1)
        cols_raw = torch.arange(self.grid_dim).view(1, -1)
        
        # Mask broadcast
        self.qubit_mask = (rows_raw + cols_raw) % 2 != 0 # Qubit: True; Syndrome: False
        self.register_buffer('qubit_indices', grid_indices[self.qubit_mask])

        rows_expanded = rows_raw.expand(self.grid_dim, self.grid_dim)
        cols_expanded = cols_raw.expand(self.grid_dim, self.grid_dim)

        y_coords = rows_expanded[self.qubit_mask]
        x_coords = cols_expanded[self.qubit_mask]
        
        coords = torch.stack([y_coords, x_coords], dim=1) 
        self.register_buffer('qubit_coords', coords)

    def forward(self, syndrome_flat):
        B = syndrome_flat.shape[0]
        L = self.L

        z_synd = syndrome_flat[:, :L*L].view(B, L, L)
        x_synd = syndrome_flat[:, L*L:].view(B, L, L)
        grid = torch.zeros(B, 1, 2*L, 2*L, device=syndrome_flat.device)
        grid[:, 0, 0::2, 0::2] = x_synd # Hàng chẵn, cột chẵn
        grid[:, 0, 1::2, 1::2] = z_synd # Hàng lẻ, cột lẻ
        
        # 2. Apply Conv Kernel dấu cộng
        # Output: (B, d_model, 2L, 2L)
        # Tại mỗi điểm (i, j), nó chứa thông tin tổng hợp từ 4 ô xung quanh
        features_map = self.conv_plus(grid)
        # 3. CHỈ LẤY CÁC VỊ TRÍ LÀ QUBIT
        # Flatten không gian: (B, d_model, 4L^2)
        features_flat = features_map.flatten(2)
        
        # Lọc lấy 2L^2 vị trí qubit: (B, d_model, 2L^2)
        # indices shape: (1, 1, 2L^2) -> mở rộng cho batch
        idx = self.qubit_indices.expand(B, features_flat.size(1), -1)
        # Dùng gather để lấy đúng cột
        # Output: (B, d_model, 2L^2)
        qubit_features = torch.gather(features_flat, 2, idx)
        
        # Transpose cho Transformer: (B, 2L^2, d_model)
        qubit_features = qubit_features.transpose(1, 2)

        batch_coords = self.qubit_coords.expand(B, -1, -1)
        
        return qubit_features, batch_coords
    
class AxialRoPE2D(nn.Module):
    """
    Axial RoPE 2D đúng paper (Eq.12-13), dạng interleaved theo kênh phức:
      angles = [theta_0 * x, theta_0 * y, theta_1 * x, theta_1 * y, ...]
      xem (2j, 2j+1) là 1 số phức rồi nhân exp(i*angle)
    """
    def __init__(self, head_dim: int, base: float = 100.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
        self.head_dim = head_dim
        self.n_freq = head_dim // 4  # t = 0..d/4-1

        t = torch.arange(self.n_freq, dtype=torch.float32)
        theta_t = base ** (-t / self.n_freq)  # (d/4,)
        self.register_buffer("theta_t", theta_t)

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        """
        q,k: (B,H,N,D) real
        coords: (B,N,2) with (y,x)
        """
        B, H, N, D = q.shape
        assert D == self.head_dim, f"D mismatch: got {D}, expected {self.head_dim}"
        assert coords.shape[:2] == (B, N)

        y = coords[..., 0].to(torch.float32)  # (B,N)
        x = coords[..., 1].to(torch.float32)  # (B,N)

        # (B,N,d/4)
        ang_x = torch.einsum("bn,t->bnt", x, self.theta_t)
        ang_y = torch.einsum("bn,t->bnt", y, self.theta_t)

        # interleave -> (B,N,d/2)
        angles = torch.stack([ang_x, ang_y], dim=-1).reshape(B, N, D // 2)

        # expand head dim -> (B,1,N,d/2)
        angles = angles.unsqueeze(1)

        # q,k -> complex (B,H,N,d/2)
        q_dtype = q.dtype
        k_dtype = k.dtype

        q2 = q.to(torch.float32).contiguous().reshape(B, H, N, D // 2, 2)
        k2 = k.to(torch.float32).contiguous().reshape(B, H, N, D // 2, 2)
        q_c = torch.view_as_complex(q2)  # complex64
        k_c = torch.view_as_complex(k2)

        # rot = exp(i*angles) complex (B,1,N,d/2)
        rot = torch.exp(1j * angles)  # complex64

        q_c = q_c * rot
        k_c = k_c * rot

        # back to real (B,H,N,D)
        q_out = torch.view_as_real(q_c).reshape(B, H, N, D).to(q_dtype)
        k_out = torch.view_as_real(k_c).reshape(B, H, N, D).to(k_dtype)
        return q_out, k_out

    
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Module RoPE 2D (tính cho từng head dimension)
        self.rotary = AxialRoPE2D(self.head_dim)
    def forward(self, x, coords, mask=None):
        """
        x: (Batch, Seq_Len, d_model)
        coords: (Batch, Seq_Len, 2)
        """
        B, N, D = x.shape
        
        # 1. Linear Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape & Transpose cho Multi-head
        # (B, N, H, D_h) -> (B, H, N, D_h)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. [QUAN TRỌNG] Apply RoPE 2D
        q,k = self.rotary(q, k, coords)
        
        # 4. Scaled Dot-Product Attention (Dùng hàm tối ưu của PyTorch 2.0)
        # Output: (B, H, N, D_h)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0)
        
        # 5. Gom lại (Concat heads)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(attn_out)
    
class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feed_forward, dropout, use_moe = False):
        super(RoPEEncoderLayer, self).__init__()
        self.d_model = d_model

        # MoE
        self.use_moe = use_moe

        # attention
        self.attn = RoPEMultiheadAttention(
            d_model, num_heads, dropout=dropout
        )

        # SoftMoE layer
        if use_moe:
            self.moe_layer = SoftMoE(
                dim = d_model,
                num_experts = 8,
                slot_per_expert = 4,
                expert_mult = 4,
                dropout = 0.,
                geglu = False,
                use_layernorm = False,
                is_dynamic = False
            )

        self.feed_forward = feed_forward
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords, mask=None): 
        # Pass coords vào attention
        attn_out = self.attn(self.ln_1(x), coords, mask)
        x = x + self.dropout(attn_out)

        if self.use_moe:
            h = self.ln_2(x)
            moe_out, slots = self.moe_layer(h)
            o_loss = self.moe_layer.compute_ortho_loss_1(slots)
            x = x + self.dropout(moe_out)
            return x, o_loss
        else:
            x = x + self.dropout(self.feed_forward(self.ln_2(x)))
            return x, None
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, feed_forward, dropout, N):
        super(Encoder, self).__init__()
        layers = []
        for i in range(N):
            use_moe = (i >= N - 2)
            layer = RoPEEncoderLayer(
                d_model,
                num_heads,
                copy.deepcopy(feed_forward),
                dropout,
                use_moe=use_moe,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(layer.d_model)
        if N > 1:
            self.norm2 = nn.LayerNorm(layer.d_model)

    def forward(self, x,coords, mask):
        total_o_loss = 0.0
        num_moe_layers = 0
        for idx, layer in enumerate(self.layers, start=1):
            x, o_loss = layer(x, coords, mask)
            
            if o_loss is not None:
                total_o_loss = total_o_loss + o_loss
                num_moe_layers += 1
            
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        
        if num_moe_layers > 0:
            avg_o_loss = total_o_loss / num_moe_layers
        else:
            avg_o_loss = None
        
        return self.norm(x), avg_o_loss
    



############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        self.no_g = args.no_g
        code = args.code
        self.s = code.pc_matrix.size(0)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)
        
        self.decoder = Encoder(
            d_model = args.d_model,
            num_heads = args.h,
            feed_forward = ff,
            dropout = dropout,
            N = args.N_dec,
        )

        if self.no_g:
            self.vit_embed = ToricViTEmbedding(
                L = args.code_L,
                d_model = args.d_model
            )
            coords = self.vit_embed.qubit_coords              # (N,2) buffer
            qec_mask = self.build_qec_mask_from_H(code.pc_matrix, coords, args.code_L, k_hop= 2)
            self.register_buffer("qec_attn_mask", qec_mask)   # (N,N)

            N = self.vit_embed.num_qubits  # = 2L^2
            assert self.qec_attn_mask.shape == (N, N)
            assert self.qec_attn_mask.dtype == torch.bool

        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])

        self.out_fc_synd = nn.Linear(code.pc_matrix.size(0), code.n) 

        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, syndrome):
        if self.no_g:
            emb, coords = self.vit_embed(syndrome) 
            mask = self.qec_attn_mask
            emb, o_loss  = self.decoder(emb, coords, mask)
            
            feats = self.oned_final_embed(emb).squeeze(-1)
            output = self.out_fc_synd(feats)
            return output, o_loss

    def loss(self, z_pred, z2):
        #dealing with DP activations
        #. loss 1 
        loss1 = F.binary_cross_entropy_with_logits(z_pred, 1-z2)

        if self.no_g:
            return loss1, None
        
    @staticmethod
    def build_qec_mask_from_H(code_pc_matrix: torch.Tensor,
                          qubit_coords: torch.Tensor,
                          L: int,
                          k_hop: int = 2):
        """
        code_pc_matrix: (s, n) long/bool, với n = 4L^2 (full_H) hoặc 2L^2
        qubit_coords: (N=2L^2, 2) coords (y,x) đúng theo token order trong model
        return: attn_mask bool (N,N), True = bị mask (cấm attend)
        """
        device = qubit_coords.device
        H = code_pc_matrix.to(device).bool()
        n_phys = 2 * L * L

        # 1. Gộp các kênh lỗi (X, Z) về cùng một vị trí vật lý
        # H matrix thường có dạng Block Diagonal: [H_Z | 0 ]
        #                                         [ 0  | H_X]
        # Cột 0 và cột n_phys cùng trỏ vào Qubit vật lý số 0.
        if H.shape[1] == 2 * n_phys:
            H_phys = H[:, :n_phys] | H[:, n_phys:]   # (s, n_phys)
        elif H.shape[1] == n_phys:
            H_phys = H
        else:
            raise ValueError(f"Unexpected H shape {H.shape}, expected (*,{n_phys}) or (*,{2*n_phys})")

        # 2) Map token -> phys_id theo đúng ToricCode indexing
        y = qubit_coords[:, 0].long()
        x = qubit_coords[:, 1].long()

        col = torch.empty_like(y)
        even = (y % 2 == 0)
        col[even]  = (x[even] - 1) // 2
        col[~even] = x[~even] // 2
        phys_id = y * L + col                              # (N,)
        # sanity
        assert phys_id.min() >= 0 and phys_id.max() < n_phys

        # 3) Build adjacency A_phys: share at least one stabilizer
        A_phys = torch.eye(n_phys, dtype=torch.bool, device=device)
        # mỗi row chạm 4 qubits -> nối clique 4 nodes
        for r in range(H_phys.shape[0]):
            idx = torch.where(H_phys[r])[0]               # thường size 4
            if idx.numel() <= 1:
                continue
            A_phys[idx[:, None], idx[None, :]] = True

        # 4) k-hop (tuỳ chọn): cho phép lan truyền xa hơn mà vẫn “đúng theo graph”
        # k_hop=1 là share-stabilizer trực tiếp
        if k_hop > 1:
            A = A_phys.clone()
            A_int = A_phys.int()
            for _ in range(k_hop - 1):
                # reachability: A = A OR (A @ A_phys)
                A_int = (A_int @ A_phys.int() > 0).int()
                A = A | (A_int > 0)
            A_phys = A
            A_phys.fill_diagonal_(True)

        # 5) Permute A_phys -> A_token theo token order
        A_tok = A_phys[phys_id][:, phys_id]               # (N,N)

        # 6) Convert to SDPA bool mask: True = masked out
        attn_mask = ~A_tok
        attn_mask.fill_diagonal_(False)
        return attn_mask


############################################################
############################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch DQEC')

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric'])
    parser.add_argument('--code_L', type=int, default=3,help='Lattice length')
    parser.add_argument('--noise_type', type=str,default='depolarization', choices=['independent','depolarization'],help='Noise model')

    # model args
    parser.add_argument('--N_dec', type=int, default=6,help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128,help='QECCT dimension')
    parser.add_argument('--h', type=int, default=8,help='Number of heads')

    # qecc args
    parser.add_argument('--use_o_loss', type=bool, default=True, help="Use orthogonal loss or not")
    parser.add_argument('--lambda_loss_ber', type=float, default=0.5,help='BER loss regularization')
    parser.add_argument('--lambda_loss_ler', type=float, default=1.,help='LER loss regularization')
    parser.add_argument('--lambda_loss_n_pred', type=float, default=0.5,help='g noise prediction regularization')
    parser.add_argument('--lambda_ortho_loss', type=float, default=0.05,help='orthogonal loss')

    # ablation args
    parser.add_argument('--no_g', type=int, default=1)
    parser.add_argument('--no_mask', type=int, default=0)

    args = parser.parse_args()

    # Tạo code object giống như trong Main.py
    class Code():
        pass
    
    code = Code()
    full_H=(args.noise_type == 'depolarization')
    H, Lx = eval(f'Get_{args.code_type}_Code')(args.code_L, full_H=full_H)
    code.logic_matrix = torch.from_numpy(Lx).long()  # (4, 4L^2)
    code.pc_matrix = torch.from_numpy(H).long()  # (2L^2, 4L^2)
    print(f"pc matrix shape: {code.pc_matrix[0]}")
    code.n = code.pc_matrix.shape[1]  # 4L^2
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    
    # Khởi tạo model
    model = ECC_Transformer(args, dropout=0)
    magnitude = torch.randn(1, code.n) # 4L^2 --> dùng để xấp xỉ error
    syndrome = torch.randint(0, 2, (1, code.pc_matrix.size(0))).float()
    print("H matrix first row (Stabilizer 0 checks which qubits?):")
    print(torch.nonzero(code.pc_matrix[0]).flatten())
    print(f"syndrome: {syndrome}")
    print(f"magnitude shape: {magnitude.shape}, syndrome shape: {syndrome.shape}")
    model(syndrome)