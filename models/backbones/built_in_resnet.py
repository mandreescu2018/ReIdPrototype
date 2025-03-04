"""Class for the ResNet and IBN-Net based feature map
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.2
        July 4, 2021
    """

from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

fea_dims_small = {'layer2': 128, 'layer3': 256, 'layer4': 512}
fea_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}

import torch
import torchvision.models as models

class ResNetFactory:
    """
    Factory class to create different versions of ResNet dynamically, handling weights correctly.
    """

    _resnet_versions = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152
    }

    _default_weights = {
        18: models.ResNet18_Weights.DEFAULT,
        34: models.ResNet34_Weights.DEFAULT,
        50: models.ResNet50_Weights.DEFAULT,
        101: models.ResNet101_Weights.DEFAULT,
        152: models.ResNet152_Weights.DEFAULT
    }

    @staticmethod
    def get_resnet(depth, weights="default", num_classes=1000):
        """
        Returns a ResNet model with the specified depth and weights.

        Args:
            depth (int): ResNet depth (18, 34, 50, 101, 152).
            weights (str or torchvision.models.Weights): 
                - "default" (uses torchvision's recommended weights)
                - None (random initialization)
                - Specific weights object (e.g., models.ResNet50_Weights.IMAGENET1K_V1)
            num_classes (int): Number of output classes (default: 1000).

        Returns:
            torch.nn.Module: ResNet model.
        """
        if depth not in ResNetFactory._resnet_versions:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Choose from {list(ResNetFactory._resnet_versions.keys())}")

        # Determine weights
        if weights == "default":
            weights = ResNetFactory._default_weights[depth]
        elif weights is None:
            weights = None  # No pretrained weights
        elif isinstance(weights, str):
            raise ValueError(f"Invalid weights argument: {weights}. Use 'default', None, or a torchvision Weights object.")

        # Create model
        model = ResNetFactory._resnet_versions[depth](weights=weights)

        # Modify classifier for different num_classes
        if num_classes != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        return model



class ResNet_BuiltIn(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, cfg, ibn_type=None, final_layer='layer3', neck=128, pretrained=True):
        super(ResNet_BuiltIn, self).__init__()

        self.depth = depth
        self.final_layer = final_layer
        self.neck = neck
        self.pretrained = pretrained

        if depth not in ResNet_BuiltIn.__factory:
            raise KeyError("Unsupported depth: ", depth)
        if ibn_type is not None and depth == 152:
            raise KeyError("Unsupported IBN-Net depth: ", depth)

        if ibn_type is None:
            # Construct base (pretrained) resnet
            print('\nCreate ResNet model ResNet-%d.\n' % depth)
            self.base = ResNet_BuiltIn.__factory[depth](pretrained=pretrained)
        else:
            # Construct base (pretrained) IBN-Net
            model_name = 'resnet%d_ibn_%s' % (depth, ibn_type)
            print('\nCreate IBN-Net model %s.\n' % model_name)
            self.base = torch.hub.load('XingangPan/IBN-Net', model_name, pretrained=pretrained)

        # if depth < 50:
        #     out_planes = fea_dims_small[final_layer]
        # else:
        #     out_planes = fea_dims[final_layer]

        out_planes = cfg.SOLVER.FEATURE_DIMENSION

        if neck > 0:
            self.neck_conv = nn.Conv2d(out_planes, neck, kernel_size=3, padding=1)
            out_planes = neck

        self.num_features = out_planes

    def forward(self, inputs):
        x = inputs
        for name, module in self.base._modules.items():
            x = module(x)
            if name == self.final_layer:
                break

        if self.neck > 0:
            x = self.neck_conv(x)

        x = F.normalize(x)

        return x



