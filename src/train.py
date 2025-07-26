import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb
import numpy as np
from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import NpySpectrogramDataset, PngSpectrogramDataset
from model import GravitySpyCNN

from data_loader import CropSpectrogramPlot

def get_dataloader(dataset_type, split, root, batch_size):
    transform = transforms.Compose([
        CropSpectrogramPlot(crop_box=(100, 70, 720, 535)) if dataset_type == "png" else lambda x: x,
        transforms.Resize((128, 128)),
        transforms.ToTensor() if dataset_type == "png" else lambda x: x,
    ])

    if dataset_type == "png":
        dataset = PngSpectrogramDataset(f"{root}/png/{split}", transform=transform)
    elif dataset_type == "npy":
        dataset = NpySpectrogramDataset(f"{root}/npy/{split}", transform=transform)
    else:
        raise ValueError("Unsupported dataset type")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True)

def plot_and_log_confusion_matrix(y_true, y_pred, class_names, output_dir, epoch):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_epoch_{epoch}.png")
    plt.close()

    wandb.log({"confusion_matrix": wandb.Image(f"{output_dir}/confusion_matrix_epoch_{epoch}.png")})


def train(args):
    wandb.init(project=args.project_name, config=vars(args), name=args.run_name)

    train_loader = get_dataloader(args.dataset_type, "train", args.root, args.batch_size)
    val_loader = get_dataloader(args.dataset_type, "val", args.root, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GravitySpyCNN(input_shape=(1, 128, 128), num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    class_names = sorted(os.listdir(f"{args.root}/{args.dataset_type}/train"))  # Get class names from the train directory
    class_names = [name for name in class_names if os.path.isdir(os.path.join(f"{args.root}/{args.dataset_type}/train", name))] # filter out non-directory entries
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for epoch in trange(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Save predictions and labels
        np.save(f"{wandb.run.dir}/val_preds_epoch_{epoch+1}.npy", np.array(all_preds))
        np.save(f"{wandb.run.dir}/val_labels_epoch_{epoch+1}.npy", np.array(all_labels))

        # Confusion matrix
        plot_and_log_confusion_matrix(all_labels, all_preds, class_names, output_dir=wandb.run.dir, epoch=epoch+1)


        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", type=str, choices=["png", "npy"], required=True)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--project-name", type=str, default="glitch-classifier")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

