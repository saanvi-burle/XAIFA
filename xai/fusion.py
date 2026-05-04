import numpy as np

def normalize(x):
    return (x - x.min()) / (x.max() + 1e-8)

def fuse_maps(maps, weights=None):

    maps = [normalize(m) for m in maps]

    if weights is None:
        weights = [1/len(maps)] * len(maps)

    fused = sum(w*m for w, m in zip(weights, maps))

    return normalize(fused)