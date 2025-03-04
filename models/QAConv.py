import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.built_in_resnet import ResNet_BuiltIn, ResNetFactory
# from .backbones.resnet_BoT_backbone import ResNet
from utils.weight_utils import weights_init_kaiming, weights_init_classifier

class QAConvBuilder(nn.Module):
    
    def __init__(self, cfg):
        last_stride=1
        depth = 50
        neck = 128
        ibn_type = 'b'
        self.final_layer = 'layer3'
        pretrained = True

        self.neck = neck

        super(QAConvBuilder, self).__init__()

        if ibn_type is None:
            # Construct base (pretrained) resnet
            print('\nCreate ResNet model ResNet-%d.\n' % depth)
            self.base = ResNetFactory.get_resnet(depth=depth)
        else:
            # Construct base (pretrained) IBN-Net
            model_name = f"resnet{depth}_ibn_{ibn_type}"
            print('\nCreate IBN-Net model %s.\n' % model_name)
            self.base = torch.hub.load('XingangPan/IBN-Net', model_name, pretrained=pretrained)

            # parser.add_argument('--final_layer', type=str, default='layer3', choices=['layer2', 'layer3', 'layer4'],
            #             help="the final layer, default: layer3")
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

   
# Query-Adaptive Convolution (QAConv)
class QAConv(nn.Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
        """
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.reset_parameters()

    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, prob_fea, gal_fea):
        hw = self.height * self.width
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)
        score = torch.einsum('p c s, g c r -> p g r s', prob_fea, gal_fea)
        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1)

        score = score.view(-1, 1, hw)
        score = self.bn(score).view(-1, hw)
        score = self.fc(score)
        score = score.view(-1, 2).sum(dim=1, keepdim=True)
        score = self.logit_bn(score)
        score = score.view(prob_size, gal_size)

        return score