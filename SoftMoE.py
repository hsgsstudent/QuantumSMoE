import math
import torch 
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helper function
def exists(val) -> bool:
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple

    if m.is_integer():
        return False, tensor
    
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# norm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class RMSNorm(Module):
    #. RMS(x) = sqrt(1/d) * l2norm
    #. RMSNorm(x) = \gamma * x/RMS
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma
    
# expert
def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

class GEGLU(Module):
    def forward(self, x):
        x, gate =  x.chunk(2, dim = -1)
        return x * F.gelu(gate)
    
def GLUFeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2/3)
    return nn.Sequential(
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim)
        )

# main class
class SoftMoE(Module):
    """
    Note:

    einstein notation
    - b: batch
    - n: sequence length (seq_len)
    - e: number of experts (num_experts)
    - s: number of slots per expert (slot_per_expert)
    - d: feature dimension
    """
    def __init__(
        self,
        dim,
        *,
        num_experts = 8,
        slot_per_expert = 8,
        expert_mult = 4,
        dropout = 0.,
        geglu = False,
        use_layernorm = False,
        is_dynamic = False
    ):
        super().__init__()
        self.norm = LayerNorm if use_layernorm else RMSNorm(dim)
        self.slot_norm = LayerNorm if use_layernorm else RMSNorm(dim)

        self.slot_embeds = \
        nn.Sequential(
            nn.Linear(dim, dim * num_experts, bias = False),
            Rearrange('b n (e d) -> b e n d', e = num_experts),
            RMSNorm(dim)
        ) if is_dynamic    \
        else nn.Parameter(torch.randn(num_experts, slot_per_expert, dim))

        expert_class = GLUFeedForward if geglu else FeedForward

        self.experts = nn.ModuleList([
            expert_class(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
        ])
        self.last_logits = None
        self.last_dispatch_weights = None
        self.last_combine_weights = None

    # tất cả slots vuông góc với nhau
    def compute_ortho_loss_0(self, slots):
            #. slots (b e s d)
            slots = rearrange(slots, 'b e s d -> b (e s) d')
            slots = F.normalize(slots, p=2, dim=-1)

            #. Tính ma trận cosine es*es
            #. Diagonal là cosine của chính nó nên cần xoá bỏ diagonal
            #. Mục tiêu giảm non-diagonal thành 0
            M = torch.matmul(slots, slots.transpose(-1, -2))

            I = torch.eye(M.size(-1), device=slots.device, dtype=slots.dtype)

            #. Mean square
            loss = torch.square(M - I).mean()
            return loss

    # tất cả slots khác experts vuông góc với nhau
    # đang ngon nhất
    def compute_ortho_loss_1(self, slots):
            #. slots (b e s d)
            b, e, s, d = slots.shape
            slots = rearrange(slots, 'b e s d -> b (e s) d')
            slots = F.normalize(slots, p=2, dim=-1)

            #. Tính ma trận cosine es*es
            #. Diagonal là cosine của chính nó nên cần xoá bỏ diagonal
            #. Mục tiêu giảm non-diagonal thành 0
            similarity = torch.matmul(slots, slots.transpose(-1, -2))
            I = torch.eye(similarity.size(-1), device=slots.device, dtype=slots.dtype)
            similarity = similarity - I

            expert_indices = torch.arange(e, device=slots.device).repeat_interleave(s)  # [0,0,...,1,1,...,e-1,e-1,...]
            mask = expert_indices.unsqueeze(0) != expert_indices.unsqueeze(1)  # (e*s, e*s)
            mask = mask.unsqueeze(0).expand(b, -1, -1)  # (b, e*s, e*s)

            inter_expert_similarity = similarity * mask.float()

            #. Mean square
            loss = torch.square(inter_expert_similarity).mean()

            return loss
    
    # trung bình slots trong 1 experts vuông góc với nhau
    # chưa thử
    def compute_ortho_loss_2(self, slots):
        #. slots (b e s d)
        expert_repr = slots.mean(dim=2)

        expert_repr = F.normalize(expert_repr, p = 2, dim = -1)
        #. Tính ma trận cosine es*es
        #. Diagonal là cosine của chính nó nên cần xoá bỏ diagonal
        #. Mục tiêu giảm non-diagonal thành 0
        M = torch.matmul(expert_repr, expert_repr.transpose(-1, -2))

        I = torch.eye(M.size(-1), device=slots.device, dtype=slots.dtype)

        #. Mean square
        loss = torch.square(M - I).mean()
        return loss
    
    # tất cả slots vuông góc với nhau dùng gram schmidt
    def compute_ortho_loss_gram_schmidt(self, slots):
        #. slots (b e s d)
        slots = rearrange(slots, 'b e s d -> b (e s) d')
        slots = F.normalize(slots, p=2, dim=-1)

        total_loss = 0.0
        for i in range(slots.size(1)):
            # Select all experts except the i-th one
            other_slots = torch.cat([slots[:, :i, :], slots[:, i+1:, :]], dim=1)
            
            # Compute orthogonal basis for other experts (Gram-Schmidt)
            u = []  # will store orthonormal basis
            for j in range(other_slots.size(1)):
                v = other_slots[:, j, :]
                for u_vec in u:
                    #. (u_vec * v).sum(dim=-1, keepdim=True) nhân vô hướng
                    #. proj = <u, v>/<u, u> * u_vec
                    proj = ((u_vec * v).sum(dim=-1, keepdim=True)) * u_vec
                    #. gram schimdt
                    v = v - proj
                #. normalize
                v_norm = torch.norm(v, dim=-1, keepdim=True)
                v = v / (v_norm + 1e-6)
                u.append(v)
            
            # Compute projection of current expert on this basis
            current_slot = slots[:, i, :]
            proj_loss = 0.0
            for u_vec in u:
                #. công thức trong bài
                proj = ((u_vec * current_slot).sum(dim=-1, keepdim=True)) * u_vec
                proj_loss = proj_loss + (proj**2).sum(dim=-1).mean()
            
            total_loss = total_loss + proj_loss
        
        return total_loss / slots.size(1)

    def forward(self, x, mask = None):
        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)
        #. X
        x = self.norm(x)
        #. Phi
        slot_embeds = self.slot_norm(self.slot_embeds)
        #. X * \Phi
        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        #. D
        # epsilon = 1e-10
        # dispatch_weights = logits.sigmoid() / (logits.sigmoid().sum(dim = 1, keepdim = True) + epsilon)
        dispatch_weights = logits.softmax(dim = 1)

        #. C
        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        self.last_logits = logits.detach()
        self.last_dispatch_weights = dispatch_weights.detach()
        self.last_combine_weights = combine_weights.detach()

        # derive slots by weighted average of input tokens using the dispatch weights from above
        #. X~
        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # route the slots per expert to each expert
        #. Y~
        out = []
        for slots_per_expert, expert in zip(slots.unbind(dim=1), self.experts):
            out.append(expert(slots_per_expert))

        out = torch.stack(out, dim=1)
        # combine back out
        out = rearrange(out, 'b e s d -> b (e s) d')

        #. Y
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        
        return out, slots