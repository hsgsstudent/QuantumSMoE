import argparse
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Codes import Get_toric_Code
from Main import QECC_Dataset, set_seed


class Code:
    pass


def _to_bin01(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    if t.numel() > 0 and t.min().item() < 0:  # if {-1,+1}
        t = (1 - t) / 2
    return t.to(torch.int64)


def split_syndrome(syndrome_1d: torch.Tensor, L: int, full_H: bool):
    s = _to_bin01(syndrome_1d.flatten())
    if full_H:
        z = s[: L * L].view(L, L).numpy()                  # Z-type stabilizers
        x = s[L * L : 2 * L * L].view(L, L).numpy()        # X-type stabilizers
    else:
        z = np.zeros((L, L), dtype=np.int64)
        x = s[: L * L].view(L, L).numpy()
    return x, z


def _idx_to_grid_pos(idx: int, L: int) -> Tuple[int, int]:
    """
    Map idx in [0, 2L^2) (row-major over (2L, L)) onto (2L, 2L) grid.
    row even  -> horizontal edge qubit at (row, 2*col+1)
    row odd   -> vertical edge qubit   at (row, 2*col)
    """
    row = idx // L          # 0..2L-1
    col = idx % L           # 0..L-1
    if (row % 2) == 0:
        return row, 2 * col + 1
    return row, 2 * col


def build_qubit_error_map(z_1d: torch.Tensor, L: int, full_H: bool) -> Dict[Tuple[int, int], str]:
    """
    Return mapping (grid_r, grid_c) -> color for data-qubit Pauli error.
    """
    zbin = _to_bin01(z_1d.flatten()).numpy().astype(np.int64)

    n_qubits = 2 * L * L

    if full_H:
        if zbin.shape[0] != 2 * n_qubits:
            raise ValueError(f"Expected z length {2*n_qubits} for depolarization/full_H, got {zbin.shape[0]}")
        x_part = zbin[:n_qubits]
        z_part = zbin[n_qubits:]
    else:
        if zbin.shape[0] != n_qubits:
            raise ValueError(f"Expected z length {n_qubits} for non-full_H, got {zbin.shape[0]}")
        x_part = zbin
        z_part = np.zeros_like(x_part)

    err_map: Dict[Tuple[int, int], str] = {}
    for idx in range(n_qubits):
        xerr = int(x_part[idx])
        zerr = int(z_part[idx])

        if xerr == 0 and zerr == 0:
            continue

        if xerr == 1 and zerr == 0:
            color = "orange"       # X error
        elif xerr == 0 and zerr == 1:
            color = "lightgreen"   # Z error
        else:
            color = "purple"       # Y error (X+Z)

        pos = _idx_to_grid_pos(idx, L)
        err_map[pos] = color

    return err_map


def plot_toric(
    L, x_syn, z_syn, err_map,
    save=None,
    show=True,
    overlay=None,
    overlay_alpha=0.55,
    overlay_cmap="magma",
    canvas=None,                 # NEW: (2L,2L) array/tensor
    canvas_alpha=0.85,           # NEW
    canvas_cmap="magma",         # NEW
):
    N = 2 * L
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect("equal")
    ax.axis("off")

    if canvas is not None:
        import numpy as np
        if hasattr(canvas, "detach"):
            canvas_np = canvas.detach().cpu().numpy()
        else:
            canvas_np = np.array(canvas)
        canvas_np = canvas_np[::-1, :]  # flip để khớp hệ tọa độ plot_toric

        canvas_im = ax.imshow(
            canvas_np,
            cmap=canvas_cmap,
            origin="lower",
            extent=[-0.5, (2*L) - 0.5, -0.5, (2*L) - 0.5],
            alpha=canvas_alpha,
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )

    # grid lines
    for r in range(N):
        ax.plot([0, N - 1], [r, r], color="black", lw=1, alpha=0.5, zorder=1)
        ax.plot([r, r], [0, N - 1], color="black", lw=1, alpha=0.5, zorder=1)

    if overlay:
        xs, ys, cs = [], [], []
        for (r, c), w in overlay.items():
            xs.append(c)
            ys.append((N - 1) - r)
            cs.append(w)
        overlay_sc = ax.scatter(xs, ys, c=cs, cmap=overlay_cmap, s=260, alpha=overlay_alpha,
                        edgecolors="none", zorder=2.5)

    # nodes
    for r in range(N):
        for c in range(N):
            y = (N - 1) - r
            x = c

            if (r + c) % 2 == 1:
                # physical data qubit
                base = "white"
                fc = err_map.get((r, c), base)
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

    mappable = None
    if canvas is not None:
        mappable = canvas_im
    elif overlay:
        mappable = overlay_sc

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_L", type=int, default=6)
    ap.add_argument("--noise_type", type=str, default="depolarization", choices=["independent", "depolarization"])
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    full_H = (args.noise_type == "depolarization")
    H, Lx = Get_toric_Code(args.code_L, full_H=full_H)

    code = Code()
    code.logic_matrix = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = "toric"

    class A:
        pass

    a = A()
    a.noise_type = args.noise_type
    a.repetitions = 1

    ds = QECC_Dataset(code, ps=[args.p], len=1, args=a)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    (_, z, _, _, syndrome) = next(iter(dl))
    syndrome_1d = syndrome[0]
    z_1d = z[0]

    x_syn, z_syn = split_syndrome(syndrome_1d, L=args.code_L, full_H=full_H)
    err_map = build_qubit_error_map(z_1d, L=args.code_L, full_H=full_H)

    plot_toric(args.code_L, x_syn, z_syn, err_map, save=args.save)


if __name__ == "__main__":
    main()