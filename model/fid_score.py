from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import numpy as np
import tensorflow as tf
import os
import cv2

def get_fid(images1, images2):

    images1 = torch.tensor(images1, dtype=torch.uint8)
    images2 = torch.tensor(images2, dtype=torch.uint8)
    
    # add batch dimension
    images1 = images1.unsqueeze(0)
    images2 = images2.unsqueeze(0)

    # batch_size*C*H*W -> batch_size*H*W*C
    images1 = images1.permute(0, 3, 1, 2)
    images2 = images2.permute(0, 3, 1, 2)

    # https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
    # float scalar tensor with mean FID value over samples
    # we keep batchsize 2 as minimum images required are 2 or more
    images1 = images1.repeat(2, 1, 1, 1)
    images2 = images2.repeat(2, 1, 1, 1)
    
    fid = FrechetInceptionDistance(feature=64)
    fid.update(images1, real=True)
    fid.update(images2, real=False)
    score = fid.compute()
    return score

# !pip install torch-fidelity