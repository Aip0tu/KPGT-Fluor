import os
import random
import warnings
import numpy as np
import torch
import dgl

def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        dgl.random.seed(seed)
        dgl.seed(seed)
    except Exception as exc:
        warnings.warn(f"Skipping DGL random seed setup: {exc}")
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            warnings.warn(f"Skipping CUDA random seed setup: {exc}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed) 


def get_device():
    """Return a usable training device, with a CPU fallback when CUDA init fails."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        warnings.warn(f"CUDA is available but initialization failed, falling back to CPU: {exc}")
        return torch.device("cpu")
