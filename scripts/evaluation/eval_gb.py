import argparse
import json
import logging
import pickle

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Eval-GB")


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate gradient boosting model.")
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("dataset_path", type=str, help="Path to dataset to evaluate for")
    parser.add_argument(
        "output_path", type=str, help="Path to save the evaluation results"
    )
    return parser


def evaluate(model, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    log.info(f"Accuracy: {accuracy}")

    roc_auc = roc_auc_score(y, y_pred)
    log.info(f"ROC AUC: {roc_auc}")

    f1 = f1_score(y, y_pred)
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
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    dataset = np.load(args.dataset_path)
    result = evaluate(model, dataset['X'], dataset['y'])

    output_path = Path(args.output_path).parent
    output_path.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f)
