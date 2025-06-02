import torch
import torch.nn as nn
import torchvision.models.vision_transformer as vits

class TorchvisionVIT(nn.Module):
    def __init__(self, model_name, pretrained=False, num_classes=1000, num_id_classes=1000):
        """
        Initialize the Vision Transformer model from torchvision.
        
        Args:
            model_name (str): Name of the Vision Transformer model.
            pretrained (bool): Whether to load pretrained weights.
            num_classes (int): Number of output classes for classification.
            num_id_classes (int): Number of identity classes for classification.
        """
        super(TorchvisionVIT, self).__init__()
        # self.model = None
        self.model = self.get_vit_model(model_name, pretrained, num_classes, num_id_classes)
    
    def get_vit_model(self, model_name, pretrained=False, num_classes=1000, num_id_classes=1000):
        """
        Get a Vision Transformer model from torchvision with specified parameters.
        """
        if model_name == 'vit_b_16':
            model = vits.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            model = vits.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            model = vits.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_l_32':
            model = vits.vit_l_32(pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        model.heads = nn.Linear(model.heads.head.in_features, model.heads.head.in_features)

        return model
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        for k, v in param_dict.items():
            if 'head' in k:
                continue
            if 'classifier' in k:
                continue

    def forward(self, x):
        """
        Forward pass through the Vision Transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)
    



