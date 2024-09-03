import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat





def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    
    *spatial_shape, _ = image_shape
   
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc



    # don't ignore the coefficient
    # An = P**2 * (2*np.pi*ns * np.sin(2*np.pi*ns) - 1 + np.cos(2*np.pi*ns)) / (4*(np.pi**2) *(ns**2)) 
    # Bn = P**2 * (np.sin(2*np.pi*ns) - 2*np.pi*ns * np.cos(2*np.pi*ns)) / (4*(np.pi**2) *(ns**2))
    # sin_emb = Bn*torch.sin(2*np.pi* (ns/P)*x)
    # print(sin_emb.shape)
    # cos_emb = An*torch.cos(2*np.pi* (ns/P)*x)
    # ignore the coefficient

def get_fourier(x, P, N):
    x = x.unsqueeze(1)
    ns = torch.arange(1,N+1).unsqueeze(0).repeat_interleave(x.shape[0],dim=0)
    sin_emb = torch.sin(2*np.pi* (ns/P)*x)
    cos_emb = torch.cos(2*np.pi* (ns/P)*x)
    return torch.hstack((sin_emb,cos_emb))

def spatial_encoding(positions,num_frequency_bands,period=1):
    '''
    takes input position tensor of shape [B,N] and creates a tensor of shape [B,2*N*num_frequency_bands]
    '''
    fpos = torch.Tensor([])
    for i in range(positions.shape[1]):
        fpos = torch.cat([fpos,get_fourier(positions[:,i], period, num_frequency_bands)],dim=-1)
    return fpos





