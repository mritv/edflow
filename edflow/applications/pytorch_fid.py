import numpy as np
import scipy.linalg as lg

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class InceptionNet(nn.Module):
    """
    Inception-Net -> used for FID and IS scores
    Code is taken directly from pytorch implementation of Inception_v3 and was slightly adapted: https://pytorch.org/docs/stable/_modules/torchvision/models/inception.html#inception_v3
    """
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False) #load weights
        
        #no grad
        for param in self.parameters():
                param.requires_grad = False
        
        self.inception.eval()
    
    def forward(self, x): #taken from original pytorch implementation
        #upsample to 299x299
        x = F.interpolate(x, (299, 299), mode="bilinear", align_corners=False)/255.0 #resizes
        x = x*2.0-1.0 #normalize
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, False)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1) #-> take this for FID
        
        # N x 1000 (num_classes)
        probs = self.inception.fc(x) #-> take this for IS
        
        return x, probs


def fid_score(mean1, cov1, mean2, cov2):
    """
    Calculates FID score from activation statistics of Inception-Net
    
    Arguments:
    ----------
    mean1 = Mean of activation statistics from set 1
    cov1 = Covariance matrix of activations from set 1
    mean2 = Mean of activation statistics from set 2
    cov2 = Covariance matrix of activations from set 2
    """
    
    covmean, temp = lg.sqrtm(gt_cov.dot(cov), disp=False)
    diff = gt_mean - mean
    fid = diff.dot(diff) + np.trace(gt_cov) + np.trace(cov) - 2*np.trace(covmean)
    
    return fid
    
