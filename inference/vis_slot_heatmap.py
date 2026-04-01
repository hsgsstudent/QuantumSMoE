import os
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

# allow import from src/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_THIS_DIR, ".."))

from Codes import Get_toric_Code
from Main import QECC_Dataset, set_seed
from Model import ECC_Transformer


class Code:
    pass


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def build_code(code_L: int, noise_type: str):
    full_H = (noise_type == "depolarization")
    H, Lx = Get_toric_Code(code_L, full_H=full_H)

    code = Code()
    code.logic_matrix = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = "toric"
    return code


def make_one_batch(code, noise_type: str, p: float, batch_size: int, device):
    class A:
        pass
    a = A()
    a.noise_type = noise_type
    a.repetitions = 1

    ds = QECC_Dataset(code, ps=[float(p)], len=batch_size, args=a)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    x, z, y, magnitude, syndrome = next(iter(dl))
    return syndrome.to(device), z.to(device)


def dispatch_weights_from_layer(moe_layer, h):
    # h: (B, N, D) at layer input to SoftMoE
    x_norm = moe_layer.norm(h)  # (B,N,D)

    # slot_embeds is Parameter (E,S,D) when is_dynamic=False
    phi = moe_layer.slot_norm(moe_layer.slot_embeds)  # (E,S,D)

    logits = torch.einsum("b n d, e s d -> b n e s", x_norm, phi)
    dispatch = logits.softmax(dim=1)  # softmax over tokens n
    return dispatch  # (B,N,E,S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--model_path",
    type=str,
    default='/Users/vietnguyen/Desktop/quantum/(run)quantum error correction code/4.Deep Quantum Error Correction/DQEC-main/src/infer/best_model.pt'
    )
    ap.add_argument("--code_L", type=int, default=4)
    ap.add_argument("--noise_type", type=str, default="depolarization", choices=["independent", "depolarization"])
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--layer_idx", type=int, default=4, help="0-based encoder layer index. Layer 5 => 4")
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "vis_slot_heatmap"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    code = build_code(args.code_L, args.noise_type)

    # build a minimal args object for ECC_Transformer
    class M:
        pass
    m = M()
    m.no_g = 1
    m.no_mask = 0
    m.code = code
    m.code_L = args.code_L
    m.noise_type = args.noise_type
    m.repetitions = 1
    m.N_dec = 6
    m.d_model = 128
    m.h = 8

    model = ECC_Transformer(m, dropout=0).to(device)
    model.eval()

    ckpt = torch.load(args.model_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    syndrome, z_true = make_one_batch(code, args.noise_type, args.p, args.batch_size, device)

    # hook lấy h = ln_2(x) output tại encoder layer args.layer_idx
    captured = {}

    target_layer = model.decoder.layers[args.layer_idx]
    assert hasattr(target_layer, "use_moe") and target_layer.use_moe, "Layer này không bật MoE (use_moe=False)."
    moe_layer = target_layer.moe_layer

    def hook_ln2(module, inputs, output):
        captured["h"] = output.detach()

    h_hook = target_layer.ln_2.register_forward_hook(hook_ln2)

    with torch.no_grad():
        _ = model(syndrome)

    h_hook.remove()

    h = captured["h"]  # (B,N,D)
    dispatch = dispatch_weights_from_layer(moe_layer, h)  # (B,N,E,S)

    coords = model.vit_embed.qubit_coords.detach().cpu()  # (N,2), y,x
    grid_dim = 2 * args.code_L

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # vẽ grid heatmap (E x S) cho sample 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B, N, E, S = dispatch.shape
    b0 = 0

    fig, axes = plt.subplots(E, S, figsize=(3 * S, 3 * E), squeeze=False)
    yy = coords[:, 0].long()
    xx = coords[:, 1].long()

    for e in range(E):
        for s in range(S):
            w = dispatch[b0, :, e, s].detach().cpu()  # (N,)
            heat = torch.zeros((grid_dim, grid_dim), dtype=torch.float32)
            heat[yy, xx] = w
            ax = axes[e][s]
            im = ax.imshow(heat.numpy(), cmap="magma")
            ax.set_title(f"e{e}-s{s}")
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / f"layer{args.layer_idx+1}_p{args.p:.3f}_dispatch_heatmaps.png", dpi=200)
    plt.close(fig)

    print("Saved:", out_dir / f"layer{args.layer_idx+1}_p{args.p:.3f}_dispatch_heatmaps.png")


if __name__ == "__main__":
    main()