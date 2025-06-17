import torch
from torch import nn

"""The Pedal class is a custom PyTorch loss module, likely designed for person re-identification 
tasks where features are compared against a memory of patch-level representations. 
It inherits from nn.Module and takes two parameters: scale, which controls the sharpness of the softmax-like operation, 
and k, which determines how many of the nearest negative samples are considered.
In the forward method, the function processes each part (or patch) of the input features separately. 
For each part, it computes a distance matrix (dist_map) between the current features and a set of center features, 
using a vectorized Euclidean distance calculation. 
The code then excludes the position corresponding to the current sample (to avoid trivial matches) and 
sorts the remaining distances to find the nearest negatives.
The method retrieves camera IDs and video IDs from a PatchMemory object, 
which are used to filter or analyze the negative samples further. 
The top-k nearest negatives' video IDs are collected for each part and stored in all_posvid.

The loss is computed using a log-sum-exp trick, which is numerically stable and commonly used in softmax-based losses. 
Specifically, it compares the sum over the top-k negatives to the sum over all negatives, scaled by the scale parameter. 
The loss for each part is normalized by the number of features, checked for NaNs, and accumulated. 
Finally, the total loss is averaged over all parts and returned along with the collected positive video IDs.

A subtle point is the use of .cuda() throughout, which assumes all tensors are on the GPU. 
If the model or data is on the CPU, this could cause device mismatch errors. 
Also, the code assumes that PatchMemory has camid and vid attributes, which should be tensors or lists convertible to tensors."""

class Pedal(nn.Module):

    def __init__(self, scale=10, k=10):
        super(Pedal, self).__init__()
        self.scale = scale  # controls the sharpness of the softmax-like operation
        self.k = k          # number of nearest negative samples to consider

    def forward(self, feature, centers, position, PatchMemory = None, vid=None, camid=None):
        loss = 0
        #  In the forward method, the function processes each part (or patch) of the input features separately. 
        all_posvid = []
        for p in range(feature.size(0)):
            # For each part, it computes a distance matrix (dist_map) between the current features and a set of center features,
            # using a vectorized Euclidean distance calculation.    
            part_feat = feature[p, :, :]
            part_centers = centers[p, :, :]
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                       part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            # dist_map.addmm_(1, -2, part_feat, part_centers.t())
            dist_map.addmm_(part_feat, part_centers.t(), beta=1, alpha=-2)
            
            trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
            
            neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
            
            # The method retrieves camera IDs and video IDs from a PatchMemory object, 
            # which are used to filter or analyze the negative samples further. 
            pos_camid = torch.tensor(PatchMemory.camid).cuda()
            pos_camid = pos_camid[(index[:,:self.k])]
            flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
            
            pos_vid = torch.tensor(PatchMemory.pid).cuda()
            pos_vid = pos_vid[(index[:,:self.k])]
            all_posvid.append(pos_vid)
            
            x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
            
            y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
            
            l = (-x + y).sum().div(feature.size(1))
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
            loss += l
        loss = loss.div(feature.size(0))

        return loss, all_posvid