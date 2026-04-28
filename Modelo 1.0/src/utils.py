import torch
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)