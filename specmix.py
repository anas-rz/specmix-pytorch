import random
import torch

def get_band(x, min_band_size, max_band_size, band_type, mask):
    assert band_type.lower() in ['freq', 'time'], f"band_type must be in ['freq', 'time']"
    if band_type.lower() == 'freq':
        axis = 2
    else:
        axis = 1
    band_size =  random.randint(min_band_size, max_band_size)
    mask_start = random.randint(0, x.size()[axis] - band_size) 
    mask_end = mask_start + band_size
    
    if band_type.lower() == 'freq':
        mask[:, mask_start:mask_end] = 1
    if band_type.lower() == 'time':
        mask[mask_start:mask_end, :] = 1
    return mask

def specmix(x, y, prob, min_band_size, max_band_size, max_frequency_bands=3, max_time_bands=3):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    k = random.random()
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size)
        print(batch_idx)
        mask = torch.zeros(x.size()[1:3])
        num_frequency_bands = random.randint(1, max_frequency_bands)
        for i in range(1, num_frequency_bands):
            mask = get_band(x, min_band_size, max_band_size, 'freq', mask)
        num_time_bands = random.randint(1, max_time_bands)
        for i in range(1, num_time_bands):
            mask = get_band(x, min_band_size, max_band_size, 'time', mask)
        lam = torch.sum(mask) / (x.size()[1] * x.size()[2])
        x = x * (1 - mask) + x[batch_idx] * mask
        y = y * (1 - lam) + y[batch_idx] * (lam)
        return x, y
    else:
        return x, y
