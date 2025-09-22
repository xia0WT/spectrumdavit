#v1.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPeaks(nn.Module):
    def __init__(self,
                 drop_prob: float = 0.5,
                 block_size: int = 100,
                 inplace: bool = False,
                 scale = 90,
                 threshold = 20,
                 add_noise = True,
                ):
        
        super(DropPeaks, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.inplace = inplace
        self.scale = scale
        self.threshold = threshold
        self.add_noise = add_noise
        
    def forward(self, x:torch.tensor):
        # x: B,3,N == B, [profile_xrd, xrd_x, xrd_y], N

        B, C, N = x.shape
        self.block_size = min(self.block_size, N)
        x_, peak, intensity  = x.permute(1,0,2)

        if self.threshold:
            mask = self.threshold_block(intensity, self.threshold, self.drop_prob)
        else:
            mask = self.bernoulli_block(intensity, self.drop_prob)
        
        peak_index = peak / self.scale * N
        block_mask = self.block(peak_index, mask)
        if self.inplace:
            x_.mul_(block_mask)
        else:
            x_ = x_.mul(block_mask)
        if self.add_noise:
            x_ = x_.add(noise(0, 0.3 ,x_.shape))
        return x_.unsqueeze(1)
    def max_scaler(self, x):
        c = torch.max(x, dim = 1)
        x = torch.div(x.transpose(1,0) ,c.values).transpose(1,0)
        return x

    def convex_prob(self, x, p):
        return torch.exp(p-p/x)

    def block(self, x, mask):
        src = torch.ones(x.shape)
        block_mask = torch.zeros(x.shape).scatter(1 ,x.mul(mask).long() ,src)
        block_mask = F.pad(block_mask,
                            (self.block_size // 2, self.block_size - self.block_size // 2 - 1),
                            "constant",
                            0)
        block_mask = F.max_pool1d(block_mask.to(x.dtype),
                                    kernel_size=self.block_size , stride=1)
        return block_mask
        
    def threshold_block(self, intensity, threshold, p):
        nonzero_mask = (intensity > 1e-6).long()
        threshold_mask = torch.where(intensity < threshold, nonzero_mask, 0)
        mask = torch.empty(threshold_mask.shape).bernoulli(p).mul(threshold_mask)
        return nonzero_mask.sub(mask).long()

    def bernoulli_block(self, intensity, p):
        intensity_probs = self.max_scaler(intensity)
        mask = torch.bernoulli(self.convex_prob(intensity_probs, 1-p))
        return mask

def noise(mean, std, size):
    return torch.normal(mean, std,size = size)
