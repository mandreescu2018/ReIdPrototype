import torch
import torch.nn as nn
from loss.loss_factory import LossComposer

# Define some example loss functions
reconstruction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()

# Combine these losses with specific weights
composer = LossComposer(
    loss_fns=[reconstruction_loss, classification_loss],
    weights=[0.5, 0.5]
)

# Example model outputs
output_reconstruction = torch.randn(10, 3)
target_reconstruction = torch.randn(10, 3)

output_classification = torch.randn(10, 5)
target_classification = torch.randint(0, 5, (10,))

# Compute the combined loss
loss = composer(output_reconstruction, target_reconstruction, output_classification, target_classification)
print("Combined Loss:", loss.item())
