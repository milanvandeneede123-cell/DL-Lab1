import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

FIGURES = [
    ("overfit/02_resnet_model_history.csv", "02_overfit_plot.png", "Overfit"),
    ("underfit/21_resnet_model_history.csv", "21_reduced_capacity_plot.png", "Reduced capacity"),
    ("regularized/10_resnet_model_history.csv", "10_regularized_plot.png", "Regularized"),
]


def plot_learning_curves(csv_path, out_path, label):
    history_df = pd.read_csv(csv_path)
    epochs = history_df["epoch"] + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history_df["loss"], label="Training Loss", color="green", linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], label="Validation Loss", color="red", linewidth=2)
    ax1.set_title(f"Learning Curves: Loss ({label})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(epochs, history_df["accuracy"], label="Training Accuracy", color="green", linewidth=2)
    ax2.plot(epochs, history_df["val_accuracy"], label="Validation Accuracy", color="red", linewidth=2)
    ax2.set_title(f"Learning Curves: Accuracy ({label})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="../resnet")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    for csv_rel, png, label in FIGURES:
        plot_learning_curves(os.path.join(args.base, csv_rel), os.path.join(args.out, png), label)
