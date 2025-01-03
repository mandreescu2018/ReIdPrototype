import pandas as pd
from pathlib import Path
from config import cfg
from matplotlib import pyplot as plt

def plot_map(data):
    # data = pd.read_csv(validation_data_path)
    # Plotting
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(data['epoch'], data['map'], marker='o', label='mAP')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP vs Epochs')
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_cmc(data):

    # Rank labels and their corresponding column names in the DataFrame
    ranks = list(range(1,51))
    rank_columns = [f'rank_{i}' for i in ranks]

    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # accuracies = [data[col] for col in rank_columns]  # Get the accuracies for this epoch
    # plt.plot(ranks, accuracies, marker='o', label='CMC Curve')

        # Plot each epoch's data
    for i, row in data.iterrows():
        accuracies = [row[col] for col in rank_columns]  # Get the accuracies for this epoch
        plt.plot(ranks, accuracies, marker='o', label=f'Epoch {int(row["epoch"])}')    

    # Adding labels, title, and grid
    plt.xlabel('Rank Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Rank Number')
    plt.legend(title="Curves")
    plt.xticks(ranks)  # Ensure that only these ranks are labeled
    plt.grid(True)

    # Show plot
    plt.show()


def plot_with_matplotlib(data):
    # data = pd.read_csv(validation_data_path)
    # Plotting
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(data['epoch'], data['rank_1'], marker='o', label='Rank-1')
    plt.plot(data['epoch'], data['rank_5'], marker='o', label='Rank-5')
    plt.plot(data['epoch'], data['rank_10'], marker='o', label='Rank-10')
    plt.plot(data['epoch'], data['rank_20'], marker='o', label='Rank-20')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('CMC Accuracy')
    plt.title('CMC Accuracy vs Epochs')
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="ReID Prototype Training")
    parser.add_argument(
        "--config_file", default="configurations/Trans_ReID/Market/vit_base.yml", help="path to config file", type=str
    )

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    training_data_path = Path(cfg.OUTPUT_DIR)/"training_log.csv"
    validation_data_path = Path(cfg.OUTPUT_DIR)/"validation_log.csv"

    dframe = pd.read_csv(validation_data_path)
    # plot_cmc(dframe)
    plot_map(dframe)
    # plot_with_matplotlib(dframe)
    # print(dframe.head())
    # print(dframe["map"].values)
    # dframe[["rank_1", "rank_5", "rank_10", "rank_20"]].plot()
    # plt.show()

    



    

    

    