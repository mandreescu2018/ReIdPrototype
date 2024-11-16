from processors.processor_base import ModelInputProcessor
# from loss.loss_maker import DynamicLossComposer

config = {
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'epochs': 10,
    # 'dataloader': DataLoader(your_dataset, batch_size=32, shuffle=True),
    'input_processor': {
        'input_keys': ['input1', 'input2']  # These should match keys in your dataset batch dictionary
    },
    # 'loss_composer': [
    #     {'name': 'cross_entropy', 'weight': 0.7},
    #     {'name': 'mse', 'weight': 0.3}
    # ]
}

if __name__ == '__main__':
    processor = ModelInputProcessor()
    processor.run()

    