"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
import numpy as np
import torch
from scipy.sparse import hstack, kron, eye, csr_matrix, block_diag
import itertools
import scipy.linalg


class ToricCode:
    '''
    From https://github.com/Krastanov/neural-decoder/
        Lattice:
        X00--Q00--X01--Q01--X02...
         |         |         |
        Q10  Z00  Q11  Z01  Q12
         |         |         |
        X10--Q20--X11--Q21--X12...
         .         .         .
    '''
    def __init__(self, L):
        '''Toric code of ``2 L**2`` physical qubits and distance ``L``.'''
        self.L = L
        self.Xflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where an X error occured
        self.Zflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where a  Z error occured
        self._Xstab = np.empty((L,L), dtype=np.dtype('b'))
        self._Zstab = np.empty((L,L), dtype=np.dtype('b'))

    @property
    def flatXflips2Zstab(self):
        L = self.L
        _flatXflips2Zstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatXflips2Zstab[i*L+j, (2*i  )%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j+1)%L] = 1
        return _flatXflips2Zstab

    @property
    def flatZflips2Xstab(self):
        L = self.L
        _flatZflips2Xstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+1)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+3)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j+1)%L] = 1
        return _flatZflips2Xstab

    @property
    def flatXflips2Zerr(self):
        L = self.L
        _flatXflips2Zerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatXflips2Zerr[0, (2*k+1)%(2*L)*L+(0  )%L] = 1
            _flatXflips2Zerr[1, (2*0  )%(2*L)*L+(k  )%L] = 1
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatZflips2Xerr[0, (2*0+1)%(2*L)*L+(k  )%L] = 1
            _flatZflips2Xerr[1, (2*k  )%(2*L)*L+(0  )%L] = 1
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab)
        if X:
            H.append(self.flatZflips2Xstab)
        H = scipy.linalg.block_diag(*H)
        return H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr)
        if X:
            E.append(self.flatZflips2Xerr)
        E = scipy.linalg.block_diag(*E)
        return E

##########################################################################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_toric_Code(L,full_H=False):
    toric = ToricCode(L)
    Hx = toric.H(Z=full_H,X=True)
    logX = toric.E(Z=full_H,X=True)    
    return Hx, logX


############################################
import stim
def build_dem_toric_depolarizing_from_H(H: np.ndarray, p: float) -> stim.DetectorErrorModel:
    """
    H: shape (n_det, n_fault) = (2L^2, 4L^2) when full_H=True
       First half faults correspond to X-part, second half to Z-part.
       Each column must be graphlike (<=2 detectors).
    Depolarizing on each physical qubit: P(X)=P(Y)=P(Z)=p/3.
    Y is represented as correlated combination of an X-part edge and a Z-part edge using '^',
    so each piece remains graphlike and works with enable_correlations=True.
    """
    n_det, n_fault = H.shape
    assert n_fault % 2 == 0
    nq = n_fault // 2   # number of physical qubits = 2L^2
    q = p / 3.0

    dets_of_fault = []
    for j in range(n_fault):
        dets = np.flatnonzero(H[:, j]).tolist()
        if len(dets) == 0 or len(dets) > 2:
            raise ValueError(f"Column {j} not graphlike; has {len(dets)} ones.")
        dets_of_fault.append(dets)

    lines = []
    for phys in range(nq):
        fx = phys
        fz = phys + nq

        Dx = dets_of_fault[fx]  # list length 1-2 (toric should be 2)
        Dz = dets_of_fault[fz]

        # X
        lines.append(
            f"error({q:.16g}) " + " ".join(f"D{d}" for d in Dx) + f" L{fx}"
        )
        # Z
        lines.append(
            f"error({q:.16g}) " + " ".join(f"D{d}" for d in Dz) + f" L{fz}"
        )
        # Y = correlated (X-part) ^ (Z-part), both graphlike => OK with correlations enabled
        lines.append(
            f"error({q:.16g}) "
            + " ".join(f"D{d}" for d in Dx) + f" L{fx} "
            + "^ "
            + " ".join(f"D{d}" for d in Dz) + f" L{fz}"
        )

    return stim.DetectorErrorModel("\n".join(lines) + "\n")




#############################################
if __name__ == "__main__":
    a,b = Get_toric_Code(3, full_H = False)
    print(a.shape) # 2L^2 x 4L^2  | L^2 x 2L^2 
    print(b.shape) # 4 x 4L^2     | 2 x 2L^2 
    class Code:
        pass
