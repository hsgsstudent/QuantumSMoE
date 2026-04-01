from __future__ import print_function
import argparse
import random
import os
from pathlib import Path
import logging
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data

# make sure we can import from src/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

from Codes import *

import time
##################################################################
##################################################################


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

##################################################################
class QECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.code = code
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1) # (2, 2L^2) -> (2L^2, 2)
        self.pc_matrix = code.pc_matrix.transpose(0, 1).clone().cpu()  # (L^2, 2L^2) -> (2L^2, L^2)
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.args = args
        
    def independent_noise(self,pp=None):
        pp = random.choice(self.ps) if pp is None else pp
        flips = np.random.binomial(1, pp, self.pc_matrix.shape[0])
        while not np.any(flips):
            flips = np.random.binomial(1, pp, self.pc_matrix.shape[0])
        return flips
    
    def depolarization_noise(self,pp=None):
        ## See original noise definition in https://github.com/Krastanov/neural-decoder/
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[0]//2
        def makeflips(q):
            q = q/3.
            flips = np.zeros((out_dimZ+out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips  = (2*q<=rand) & (rand<3*q)
            ###
            x_flips = rand < q
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            ###
            z_flips = (q<=rand) & (rand<2*q)
            flips[out_dimZ:out_dimZ+out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ+out_dimX] ^= both_flips
            return flips
        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp)
        return flips*1.
        
        
    
    def __getitem__(self, index):
        x = self.zero_cw
        pp = random.choice(self.ps)
        if self.args.repetitions <= 1:
            z = torch.from_numpy(self.noise_method(pp))
            y = bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(z.long(),
                                    self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome) 
            return x.float(), z.float(), y.float(), (magnitude*0+1).float(), syndrome.float()
        ###
        ### See original setting definition in https://pymatching.readthedocs.io/en/stable/toric-code-example.html# 
        qq = pp
        
        noise_new = np.stack([self.noise_method(pp) for _ in range(self.args.repetitions)],1)
        noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[:,-1]
        syndrome = (torch.matmul(torch.from_numpy(noise_cumulative).long().transpose(0,1),self.pc_matrix) % 2).transpose(0,1).numpy()
        syndrome_error = (np.random.rand(self.pc_matrix.shape[1], self.args.repetitions) < qq).astype(np.uint8)
        syndrome_error[:,-1] = 0 # Perfect measurements in last round to ensure even parity
        noisy_syndrome = (syndrome + syndrome_error) % 2
        # Convert to difference syndrome
        noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
        
        z = torch.from_numpy(noise_total)
        syndrome = bin_to_sign(torch.from_numpy(noisy_syndrome)) #TODO: check if bin2sign is needed

        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return x.float(), z.float(), (y*0+1).float(), (magnitude*0+1).float(), syndrome.float().transpose(0,1)
    
    def __len__(self):
        return self.len


##################################################################
##################################################################
class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output*(torch.abs(x[0])<=1)

def binarization(y):
    return sign_to_bin(Binarization.apply(y))
###
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2
###
def diff_GF2_mul(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x
    
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp
##################################################################

def _syndrome_for_viz(s: torch.Tensor) -> torch.Tensor:
    s = s.detach().cpu()
    if s.ndim >= 2:
        s = s[-1]  # lấy round cuối nếu repetitions>1
    return s.flatten()

def _get_module(m):
    return m.module if hasattr(m, "module") else m

def build_plus_patch_heatmap(token_weights: torch.Tensor, coords: torch.Tensor, grid_dim: int) -> torch.Tensor:
    """
    token_weights: (N,) dispatch weights for one (expert,slot)
    coords: (N,2) (y,x) on 2Lx2L grid for each token
    return: (grid_dim,grid_dim) in [0,1], white=high
    """
    w = token_weights.detach().float().cpu()
    c = coords.detach().long().cpu()
    heat = torch.zeros((grid_dim, grid_dim), dtype=torch.float32)

    for i in range(c.shape[0]):
        y = int(c[i, 0].item())
        x = int(c[i, 1].item())
        wi = float(w[i].item())

        for dy, dx in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            yy = (y + dy) % grid_dim
            xx = (x + dx) % grid_dim
            heat[yy, xx] += wi

    m = float(heat.max().item())
    if m > 0:
        heat = heat / m
    return heat

def save_slot_images_one_case(dispatch, coords, L, out_dir, case_id, x_syn, z_syn, err_map, plot_toric_fn):
    # dispatch: (B,N,E,S)
    B, N, E, S = dispatch.shape

    for e in range(E):
        for s in range(S):
            w = dispatch[0, :, e, s]  # (N,)
            canvas = stitch_plus_patches_to_image(w, coords, L)  # (2L,2L) in [0,1]
            plot_toric_fn(
                int(L),
                x_syn,
                z_syn,
                err_map,
                save=str(out_dir / f"case_{case_id}_e{e}_s{s}.png"),
                show=False,
                canvas=canvas,
                canvas_cmap="magma",
                canvas_alpha=0.85,
            )

def heatmap_to_overlay_dict(heat: torch.Tensor) -> dict:
    h = heat.detach().cpu()
    H, W = h.shape
    return {(r, c): float(h[r, c].item()) for r in range(H) for c in range(W)}

def sparsify_weights(w: torch.Tensor, keep_frac: float = 0.2) -> torch.Tensor:
    """
    Giữ lại keep_frac patch có weight lớn nhất, patch còn lại = 0 (đen).
    Output được normalize về [0,1].
    """
    w = w.detach().float().cpu()
    k = max(1, int(w.numel() * keep_frac))
    thr = torch.topk(w, k).values.min()
    w = torch.where(w >= thr, w, torch.zeros_like(w))
    m = float(w.max().item())
    return (w / m) if m > 0 else w

def compute_dispatch_from_layer5(model, syndrome):
    m = _get_module(model)
    layer5 = m.decoder.layers[4]  # layer 5 (0-based)

    # nếu last_dispatch_weights chưa được set thì chạy forward 1 lần
    if getattr(layer5.moe_layer, "last_dispatch_weights", None) is None:
        with torch.no_grad():
            _ = model(syndrome)

    dispatch = layer5.moe_layer.last_dispatch_weights  # (B,N,E,S)
    coords = m.vit_embed.qubit_coords                  # (N,2)
    return dispatch, coords

def save_dispatch_heatmaps(dispatch, coords, L, out_png_path, title_prefix=""):
    """
    dispatch: (B,N,E,S)
    coords:   (N,2) y,x on 2Lx2L grid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    B, N, E, S = dispatch.shape
    grid_dim = 2 * L

    yy = coords[:, 0].long().detach().cpu()
    xx = coords[:, 1].long().detach().cpu()

    b0 = 0
    fig, axes = plt.subplots(E, S, figsize=(3.0 * S, 3.0 * E), squeeze=False)

    for e in range(E):
        for s in range(S):
            w = dispatch[b0, :, e, s].detach().cpu()  # (N,)
            heat = torch.zeros((grid_dim, grid_dim), dtype=torch.float32)
            heat[yy, xx] = w

            ax = axes[e][s]
            ax.imshow(heat.numpy(), cmap="magma")
            ax.set_title(f"{title_prefix}e{e}-s{s}".strip())
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png_path, dpi=200)
    plt.close(fig)

def stitch_plus_patches_to_image(
    token_weights: torch.Tensor,  # (N,)
    coords: torch.Tensor,         # (N,2) (y,x)
    L: int,
) -> torch.Tensor:
    """Return image (2L,2L) where each token paints a plus (+) with its weight."""
    grid = 2 * int(L)
    w = token_weights.detach().float().cpu()
    c = coords.detach().long().cpu()

    img = torch.zeros((grid, grid), dtype=torch.float32)

    for i in range(c.shape[0]):
        y = int(c[i, 0].item())
        x = int(c[i, 1].item())
        wi = float(w[i].item())

        for dy, dx in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            yy = (y + dy) % grid   # toric wrap
            xx = (x + dx) % grid
            img[yy, xx] += wi

    m = float(img.max().item())
    if m > 0:
        img = img / m
    return img

def plot_toric_with_canvas(L: int, x_syn, z_syn, err_map, canvas: torch.Tensor, save: str, canvas_cmap="magma", canvas_alpha=0.85):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    N = 2 * L
    canvas_np = canvas.detach().cpu().numpy()
    # match plot_toric coordinate system (y = (N-1)-r)
    canvas_np = canvas_np[::-1, :]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # canvas background
    ax.imshow(
        canvas_np,
        cmap=canvas_cmap,
        origin="lower",
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
        alpha=canvas_alpha,
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )

    # grid lines
    for r in range(N):
        ax.plot([0, N - 1], [r, r], color="black", lw=1, alpha=0.5, zorder=1)
        ax.plot([r, r], [0, N - 1], color="black", lw=1, alpha=0.5, zorder=1)

    # nodes (giữ màu mè như toric plot)
    for r in range(N):
        for c in range(N):
            y = (N - 1) - r
            x = c

            if (r + c) % 2 == 1:
                fc = err_map.get((r, c), "white")   # <-- tô màu lỗi
                size = 170 if (r, c) in err_map else 140
            elif (r % 2 == 0) and (c % 2 == 0):
                # X syndrome (vertex)
                u, v = r // 2, c // 2
                fc = "limegreen" if x_syn[u, v] == 1 else "dodgerblue"
                size = 170
            else:
                # Z syndrome (plaquette)
                u, v = r // 2, c // 2
                fc = "saddlebrown" if z_syn[u, v] == 1 else "red"
                size = 170

            ax.scatter(x, y, s=size, facecolors=fc, edgecolors="black", linewidths=1.5, zorder=3)

    fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_weight_image(img: torch.Tensor, out_png: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def test(args, model, device, test_loader_list, ps_range_test, cum_count_lim=10):
    model.eval()
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    test_time = []
    t = time.time()

    record_target = int(getattr(args, "record_correct_n", 0))
    record_criterion = getattr(args, "record_correct_criterion", "logical")
    record_max_steps = int(getattr(args, "record_max_steps", 50000))
    recorded_total = 0

    full_H = (args.noise_type == "depolarization")
    out_dir = Path(getattr(args, "record_correct_out", "") or (Path(__file__).resolve().parent / "vis_correct_cases"))

    if record_target > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[record] saving {record_target} correct cases to: {out_dir}")
        import matplotlib
        matplotlib.use("Agg")  # để plot_toric không bật cửa sổ

        import importlib.util

        viz_path = (Path(__file__).resolve().parents[1] / "visualize_toric.py")  # src/visualize_toric.py
        spec = importlib.util.spec_from_file_location("dqec_src_visualize_toric", str(viz_path))
        viz = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(viz)

        split_syndrome = viz.split_syndrome
        build_qubit_error_map = viz.build_qubit_error_map
        plot_toric = viz.plot_toric

    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.0
            t_start = time.time()

            it = iter(test_loader)
            steps = 0

            while True:
                steps += 1
                if steps > record_max_steps:
                    logging.warning(f"[record] reached record_max_steps={record_max_steps}, stopping early.")
                    break

                try:
                    (x, z, y, magnitude, syndrome) = next(it)
                except StopIteration:
                    it = iter(test_loader)
                    (x, z, y, magnitude, syndrome) = next(it)

                z_true = z.to(device).round().long()
                z_pred_logits, o_loss = model(syndrome.to(device))
                z_pred = sign_to_bin(torch.sign(-z_pred_logits)).long()

                logic_matrix = test_loader.dataset.logic_matrix.to(device)

                test_ber += BER(z_pred, z_true) * z.shape[0]
                test_ler += FER(
                    logical_flipped(logic_matrix, z_pred),
                    logical_flipped(logic_matrix, z_true),
                ) * z.shape[0]
                cum_count += z.shape[0]

                # record correct samples (ưu tiên đúng theo logical)
                if record_target > 0 and recorded_total < record_target:
                    lf_pred = logical_flipped(logic_matrix, z_pred)
                    lf_true = logical_flipped(logic_matrix, z_true)
                    logical_ok = torch.all(lf_pred == lf_true, dim=1)
                    exact_ok = torch.all(z_pred == z_true, dim=1)
                    ok = logical_ok if record_criterion == "logical" else exact_ok

                    ok_idx = torch.nonzero(ok, as_tuple=False).flatten().tolist()
                    for bi in ok_idx:
                        if recorded_total >= record_target:
                            break

                        synd_1d = _syndrome_for_viz(syndrome[bi])
                        z_true_1d = z_true[bi].detach().cpu()
                        z_pred_1d = z_pred[bi].detach().cpu()

                        case_id = f"{recorded_total:03d}_p{float(ps_range_test[ii]):.3e}"
                        torch.save(
                            {
                                "L": int(args.code_L),
                                "p": float(ps_range_test[ii]),
                                "full_H": bool(full_H),
                                "syndrome": synd_1d,
                                "z_true": z_true_1d,
                                "z_pred": z_pred_1d,
                                "criterion": record_criterion,
                            },
                            out_dir / f"case_{case_id}.pt",
                        )

                        x_syn, z_syn = split_syndrome(synd_1d, L=int(args.code_L), full_H=full_H)

                        err_true = build_qubit_error_map(z_true_1d, L=int(args.code_L), full_H=full_H)
                        err_pred = build_qubit_error_map(z_pred_1d, L=int(args.code_L), full_H=full_H)

                        plot_toric(int(args.code_L), x_syn, z_syn, err_true, save=str(out_dir / f"case_{case_id}_true.png"), show=False)
                        # dispatch weights layer 5
                        dispatch, coords = compute_dispatch_from_layer5(model, syndrome.to(device))  # (B,N,E,S), (N,2)
                        # save all slots for this one case
                        save_slot_images_one_case(dispatch, coords, int(args.code_L), out_dir, case_id, x_syn, z_syn, err_true, plot_toric)
                        
                        recorded_total += 1
                        logging.info(f"[record] saved {recorded_total}/{record_target}: case_{case_id}")

                done_metric = (cum_count >= cum_count_lim)
                done_record = (record_target == 0) or (recorded_total >= record_target)
                if done_metric and done_record:
                    break

            t_end = time.time()
            delta_t = t_end - t_start
            test_time.append(delta_t)
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / max(cum_count, 1))
            test_loss_ler_list.append(test_ler / max(cum_count, 1))

            print(f"Test p={ps_range_test[ii]:.3e}, BER={test_loss_ber_list[-1]:.3e}, LER={test_loss_ler_list[-1]:.3e}")
            print(f"# Sample test time: t = {delta_t*1000:4f} ms; Avg test time per sample: t = {delta_t/max(cum_count_lim,1)*1000:4f} ms")

        logging.info("Test LER  " + " ".join([f"p={p:.2e}: {v:.2e}" for v, p in zip(test_loss_ler_list, ps_range_test)]))
        logging.info("Test BER  " + " ".join([f"p={p:.2e}: {v:.2e}" for v, p in zip(test_loss_ber_list, ps_range_test)]))
        logging.info(f"Mean LER = {np.mean(test_loss_ler_list):.3e}, Mean BER = {np.mean(test_loss_ber_list):.3e}")

    logging.info(f"# of testing samples: {cum_samples_all}\n Total test time {time.time() - t} s\n")
    return test_loss_ber_list, test_loss_ler_list

##################################################################
##################################################################
##################################################################

# Move Code class outside __main__ for Windows multiprocessing compatibility
class Code():
    pass

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    args.code.logic_matrix = args.code.logic_matrix.to(device) # (2, 2L^2)
    args.code.pc_matrix = args.code.pc_matrix.to(device) # (L^2, 2L^2)
    code = args.code
    assert 0 < args.repetitions 
    
    if args.repetitions > 1:
        from Model_T_measurements import ECC_Transformer
    else:
        from Model import ECC_Transformer

    #################################
    # Load model
    model = ECC_Transformer(args, dropout=0)
    model.to(device)
    
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    
    # Load trained weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load with strict=False to ignore missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        logging.warning(f'Missing keys in checkpoint: {missing_keys}')
    if unexpected_keys:
        logging.warning(f'Unexpected keys in checkpoint: {unexpected_keys}')
    
    model.to(device)
    model.eval()
    
    logging.info(f'Model loaded from: {args.model_path}')
    logging.info(f'Model trained at epoch {checkpoint.get("epoch", "unknown")}, loss={checkpoint.get("loss", "unknown")}')
    
    #################################
    # Test configuration

    # ps_test = np.array([args.p], dtype=float)

    # test_loader = DataLoader(
    #     QECC_Dataset(code, [float(args.p)], len=int(args.num_samples), args=args),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0
    # )

    # test(args, model, device, [test_loader], ps_test, cum_count_lim=int(args.num_samples))

    
    ps_test = np.array([0.1], dtype=float)  # chỉ test 1 điểm p=0.05

    test_dataloader_list = [
        DataLoader(
            QECC_Dataset(code, [float(ps_test[-1])], len=int(args.test_num_samples_per_p), args=args),
            batch_size=int(args.test_batch_size),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    ]

    test(args, model, device, test_dataloader_list, ps_test, cum_count_lim=int(args.cum_count_lim))
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DQEC - Inference Mode')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0=single process, 4-8=faster data loading)')
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for inference (256-512 optimal for GPU, 1 for debugging)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--model_path',
        type=str,
        default='/Users/vietnguyen/Desktop/quantum/(run)quantum error correction code/4.Deep Quantum Error Correction/DQEC-main/src/infer/best_model.pt',
        help='Path to trained model checkpoint (.pt file)',
    )
    # parser.add_argument('--p', type=float, default=0.1,
    #                     help='physical error rate used to sample 1 test point')

    # parser.add_argument('--num_samples', type=int, default=10,
    #                     help='number of samples to run (set 1 for one-shot)')
    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric'])
    parser.add_argument('--code_L', type=int, default=6,help='Lattice length (MUST match the model training L)')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='depolarization', choices=['independent','depolarization'],help='Noise model')

    # model args
    parser.add_argument('--N_dec', type=int, default=6,help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128,help='QECCT dimension')
    parser.add_argument('--h', type=int, default=8,help='Number of heads')

    # Model architecture args (needed for loading)
    parser.add_argument('--no_g', type=int, default=1, help='Disable g module')
    parser.add_argument('--no_mask', type=int, default=0, help='Disable masking')

    parser.add_argument("--cum_count_lim", type=int, default=50, help="Số sample dùng để tính BER/LER cho mỗi p")
    parser.add_argument("--test_num_samples_per_p", type=int, default=2000, help="Dataset length cho mỗi p")
    parser.add_argument("--record_correct_n", type=int, default=10, help="Record N case decode đúng để visualize (0 = tắt)")
    parser.add_argument("--record_correct_out", type=str, default="", help="Thư mục output (default: src/infer/vis_correct_cases)")
    parser.add_argument("--record_correct_criterion", type=str, default="logical", choices=["logical", "exact"])
    parser.add_argument("--record_max_steps", type=int, default=50000)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################
    code = Code()
    code_fn = globals()[f"Get_{args.code_type}_Code"]
    H, Lx = code_fn(args.code_L, full_H=(args.noise_type == "depolarization"))
    code.logic_matrix = torch.from_numpy(Lx).long() # (2, 2L^2)
    code.pc_matrix = torch.from_numpy(H).long() # (L^2, 2L^2)
    code.n = code.pc_matrix.shape[1] # 2L^2
    code.k = code.n - code.pc_matrix.shape[0] # 2L^2 - 2
    code.code_type = args.code_type
    args.code = code
    
    ####################################################################
    # Create inference results directory
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler()])

    main(args)