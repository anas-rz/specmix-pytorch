import random
import torch


def specmix(x, y, alpha, prob, f_band_size, t_band_size):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        mask_start = random.randint(0, x.size()[1] - f_band_size)        
        mask_end = mask_start + f_band_size
        x[:, mask_start:mask_end, :] = x[batch_idx, mask_start:mask_end,: ]
        lam_freq = (f_band_size / x.size()[1])
        mask_start = random.randint(0, x.size()[2] - t_band_size)
        mask_end = mask_start + t_band_size
        x[:, :, mask_start:mask_end] = x[batch_idx, :, mask_start:mask_end]
        lam_time = ((mask_end - mask_start) / x.size()[2])
        lam = 1 - (lam_freq + lam_time - (f_band_size * t_band_size) / (x.size()[1] * x.size()[2]))
        y = y * lam + y[batch_idx] * (1 - lam)
        return x, y
    else:
        return x, y
