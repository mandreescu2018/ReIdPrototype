import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt

from solver.lr_scheduler import WarmupMultiStepLR
from solver.lr_scheduler_prototype import LrScheduler
from config import cfg
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    # "configurations\Trans_ReID\Market\vit_base.yml"
    
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    

    learning_rates =[]
    # Example model and optimizer
    model = nn.Linear(10, 1)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    # Initialize the dynamic scheduler
    dynamic_lr_scheduler = LrScheduler(optimizer, cfg)

    # Simulated training loop
    for epoch in range(1, 120):
        # Simulate training step
        optimizer.zero_grad()
        loss = model(torch.randn(32, 10)).sum()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        dynamic_lr_scheduler.step(epoch)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"Epoch {epoch}, LR: {current_lr:.6f}")

    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
