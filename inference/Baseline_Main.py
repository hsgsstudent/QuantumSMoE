"""
Implementation of "Deep Quantum Error Correction" (DQEC), Viet
"""
from __future__ import print_function
import argparse
import random
import os
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
import sys
sys.path.append('..') 
from Codes import *
import time
from pymatching import Matching

##################################################################
##################################################################

class Code():
    """Simple container class for quantum error correction code parameters"""
    pass

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

##################################################################
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2

class ECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.code = code
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1) # (2,2L^2 -> (2L^2,2)
        self.pc_matrix = code.pc_matrix.transpose(0, 1) # (L^2,2L^2 -> (2L^2,L^2)
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
        x = self.zero_cw # 2L^2 codeword
        pp = random.choice(self.ps)
        if self.args.repetitions <= 1:
            z = torch.from_numpy(self.noise_method(pp)) # 2L^2 noise
            y = bin_to_sign(x) + z # z + 1 do x == 0 nên bin_to_sign(x) == 1
            magnitude = torch.abs(y) # magnitude = z + 1
            syndrome = torch.matmul(z.long(),
                                    self.pc_matrix) % 2 # z * H^T | syndrome = H * z
            syndrome = syndrome
            return x.float(), z.float(), y.float(), (magnitude*0+1).float(), syndrome.float()
        ###
        qq = pp
        ### See original setting definition in https://pymatching.readthedocs.io/en/stable/toric-code-example.html# 

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
        syndrome = torch.from_numpy(noisy_syndrome)

        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        return x.float(), z.float(), (y*0+1).float(), (magnitude*0+1).float(), syndrome.float()
    
    def __len__(self):
        return self.len


##################################################################
def test(args,model, device, test_loader_list, ps_range_test, cum_count_lim=1e6): # num trials = 1e6
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    test_time = []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.
            t_start = time.time()
            while True:
                (x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_pred = []
                for ssynd in syndrome:
                    ssynd = ssynd.numpy()
                    if args.decoder == 'u-f':
                        ssynd = ssynd.T
                        pred = model(ssynd)
                    elif args.decoder == 'mwpm-corr':
                        pred = model[ii].decode(ssynd.astype(np.uint8), enable_correlations=True)  # :contentReference[oaicite:3]{index=3}
                    elif args.decoder == 'mwpm':
                        pred = model(ssynd)
                    elif args.decoder == 'mwpm-bp':
                        pred = model[ii].decode(ssynd.astype(np.uint8))
                    elif args.decoder == 'bp-lsd':
                        pred = model[ii].decode(ssynd.astype(np.uint8))
                    z_pred.append(torch.from_numpy(pred.astype(float))) # Transpose is for U-F
                
                # Stack list of tensors into single tensor
                z_pred = torch.stack(z_pred)  # [batch_size, 2L^2]
                z_device = z.to(device)
                logic_matrix = test_loader.dataset.logic_matrix
                
                test_ber += BER(z_pred, z_device) * z.shape[0]
                test_ler += FER(
                    logical_flipped(logic_matrix, z_pred), 
                    logical_flipped(logic_matrix, z_device)
                ) * z.shape[0]
                cum_count += z.shape[0]
                if cum_count >= cum_count_lim:
                    break
            t_end = time.time()
            delta_t = t_end - t_start
            test_time.append(delta_t)
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)
            print(f'Test p={ps_range_test[ii]:.3e}, BER={test_loss_ber_list[-1]:.3e}, LER={test_loss_ler_list[-1]:.3e}')
            print(f'# Sample test time: t = {delta_t*1000:4f} ms; Avg test time per sample: t = {delta_t/cum_count_lim*1000:4f} ms')
        ###
        logging.info('Test LER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info('Test BER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, ps_range_test))]))
        logging.info(f'Mean LER = {np.mean(test_loss_ler_list):.3e}, Mean BER = {np.mean(test_loss_ber_list):.3e}')
    logging.info(f'# of testing samples: {cum_samples_all}\n Total test time {time.time() - t} s')
    logging.info(f'# test time per sample: ' + ', '.join([f'{t/cum_count_lim*1000:.4f} ms' for t in test_time]) + f'\n Avg: t = {sum(test_time)/len(test_time)/cum_count_lim*1000:.4f} ms')
    return test_loss_ber_list, test_loss_ler_list

##################################################################
##################################################################
##################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    args.code.pc_matrix = args.code.pc_matrix.to(device) 
    code = args.code
    assert 0 < args.repetitions 
    #################################
    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    if args.repetitions > 1:
        ps_test = np.linspace(0.02, 0.04, 18)
    ####
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    if args.decoder == 'u-f':
        from scipy import sparse
        from UnionFindPy import Decoder
        model = Decoder(sparse.csr_matrix(code.pc_matrix.int().numpy()),
                        repetitions=args.repetitions if args.repetitions > 1 else None)
        # UF decoder: call model(syndrome)
        model_fn = model
    elif args.decoder == 'mwpm':
        m = Matching.from_check_matrix(
            code.pc_matrix,
            repetitions=args.repetitions if args.repetitions > 1 else None
        )
        model_fn = m.decode
    elif args.decoder == 'mwpm-corr':
        assert args.noise_type == 'depolarization'
        assert args.repetitions == 1  # perfect syndrome 2D only

        H_np = code.pc_matrix.int().numpy()  # shape (2L^2, 4L^2)

        matchers = []
        for p in ps_test:
            dem = build_dem_toric_depolarizing_from_H(H_np, float(p))
            m = Matching.from_detector_error_model(dem, enable_correlations=True)
            matchers.append(m)

        model_fn = matchers
    elif args.decoder == 'mwpm-bp':
        from beliefmatching import BeliefMatching
        assert args.noise_type == 'depolarization'
        assert args.repetitions == 1  # perfect syndrome

        H_np = code.pc_matrix.int().numpy()  # (2L^2, 4L^2)
        bms = []
        for p in ps_test:
            dem = build_dem_toric_depolarizing_from_H(H_np, float(p))  # DEM có '^'
            bm = BeliefMatching(dem, max_bp_iters=20)
            bms.append(bm)
        model_fn = bms
    elif args.decoder == 'bp-lsd':
        from ldpc.bplsd_decoder import BpLsdDecoder
        assert args.repetitions == 1  # bạn đang perfect syndrome measurement

        H_ldpc = code.pc_matrix.int().numpy().astype(np.uint8)  # shape (2L^2, 4L^2)

        decoders = []
        for p in ps_test:
            # depolarizing -> marginal flip prob on X-part and Z-part bits
            q = 2.0 * float(p) / 3.0
            dec = BpLsdDecoder(
                H_ldpc,
                error_rate=q,
                bp_method='product_sum',   # hoặc 'minimum_sum'
                max_iter=30,              # docs gợi ý ~30 là thường đủ
                schedule='serial',        # thử serial thường ổn cho BP
                # NOTE: API dùng tên osd_method/osd_order dù là LSD (đúng như docs ví dụ)
                lsd_method='lsd_cs',
                lsd_order=2
            )
            decoders.append(dec)

        model_fn = decoders

    logging.info(f'Model = {model_fn}')
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    test(args, model_fn, device, test_dataloader_list, ps_test)
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='PyTorch QECCT')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')  # Changed from '-1' to '0' to use GPU by default
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric',choices=['toric','surface'])
    parser.add_argument('--code_L', type=int, default=6)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--noise_type', type=str,default='depolarization', choices=['independent','depolarization'])
    #    
    parser.add_argument('--decoder', type=str,default='mwpm-bp', choices=['u-f','mwpm','mwpm-corr','mwpm-bp', 'bp-lsd'])

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    code = Code()
    H,Lx = eval(f'Get_{args.code_type}_Code')(args.code_L,full_H=args.noise_type == 'depolarization')
    code.logic_matrix = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    ####################################################################
    model_dir = os.path.join('Data_4_paper', 'Baseline_Results_QDEC', args.decoder, args.code_type, 
                             'Code_L_' + str(args.code_L) , 
                             f'noise_model_{args.noise_type}', 
                             f'repetition_{args.repetitions}' ,
                             f'test_batch_size_{args.test_batch_size}', 
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)