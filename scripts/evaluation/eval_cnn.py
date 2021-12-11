import argparse
import json
import logging
import os
import sys

import torch

sys.path.append(os.path.join(sys.path[0], '../'))

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from training.cnn_model import CNN_Model, construct_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Eval-CNN")



def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate 3D-CNN model.")
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("dataset_path", type=str, help="Path to dataset to evaluate for")
    parser.add_argument(
        "output_path", type=str, help="Path to save the evaluation results"
    )
    return parser


def evaluate(model, dataset):
    y = dataset['y']
    dataloader = construct_dataloader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X, _ = batch
            prediction = model(X)
            predictions.append(prediction.cpu())
    predictions = torch.cat(predictions, dim=0)  # type: ignore
    predictions = torch.sigmoid(predictions).numpy() > 0.5  # type: ignore

    accuracy = accuracy_score(y, predictions)
    log.info(f"Accuracy: {accuracy}")

    roc_auc = roc_auc_score(y, predictions)
    log.info(f"ROC AUC: {roc_auc}")

    f1 = f1_score(y, predictions)
    log.info(f"F1-Score: {f1}")

    result = {
        "Accuracy": accuracy,
        "ROC_AUC": roc_auc,
        "F1-Score": f1,
    }
    return result


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    log.info(f"Evaluating model on {args.dataset_path} dataset...")

    dataset = np.load(args.dataset_path)
    n_channels = dataset['X'].shape[-1]

    parameters = {
        "lr": 0.0,
        "l2_lambda": 0.0,
        "dropout": 0.0,
        "n_channels": n_channels,
        "weight_of_ones": 0.0,
    }

    model = CNN_Model.load_from_checkpoint(args.model_path, params=parameters)
    model.eval()

    result = evaluate(model, dataset)

    output_path = Path(args.output_path).parent
    output_path.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f)
