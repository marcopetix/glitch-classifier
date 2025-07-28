# system imports
import os
import sys
import argparse

# general imports
import numpy as np
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# wandb for logging
import wandb
from tqdm import trange, tqdm

# sklearn for metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from datetime import datetime

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.models import get_model
from src.data.loaders import get_dataloader
from src.utils import * # to be implemented (visualization.py, metrics.py, wandb_utils.py)


project_name = "glitch-classifier"
task = "cls" # classification task

def train(args):
    run_name = f"{task}_{args.dataset_type}_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=project_name, config=vars(args), name=run_name)

    train_loader = get_dataloader(args.dataset_type, "train", args.root, args.batch_size)
    val_loader = get_dataloader(args.dataset_type, "val", args.root, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_name, input_shape=(1, 128, 128), num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    

def main():
    parser = argparse.ArgumentParser(description="Train a classification model on spectrogram data.")
    parser.add_argument("--dataset_type", type=str, choices=["png", "npy"], required=True, help="Type of dataset to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--model_name", type=str, choices=["gspy"], default="gspy", help="Name of the model to train.")

    args = parser.parse_args()

    train(args)