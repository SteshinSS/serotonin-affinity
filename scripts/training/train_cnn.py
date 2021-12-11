import argparse
import logging

import numpy as np
import pytorch_lightning as pl

from cnn_model import CNN_Model, construct_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Train-CNN")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Trains 3D CNN for molecules classification by their voxel grid."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to folder with train/val voxels"
    )
    parser.add_argument("output_path", type=str, help="Path to save result model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2_lambda", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=228)
    return parser


def get_weight_of_ones(labels):
    total_elements = labels.size
    total_ones = (labels == 1).sum()
    coefficient = (total_elements - total_ones) / total_ones
    return coefficient


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    train_data = np.load(args.input_path + "/train.npz")
    n_channels = train_data["X"].shape[-1]
    train_dataloader = construct_dataloader(train_data, args.batch_size, shuffle=True)

    val_data = np.load(args.input_path + "/val.npz")
    val_dataloader = construct_dataloader(val_data, args.batch_size, shuffle=False)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=15,
        deterministic=True,
        checkpoint_callback=False,
    )

    parameters = {
        "lr": args.learning_rate,
        "l2_lambda": args.l2_lambda,
        "dropout": args.dropout,
        "n_channels": n_channels,
        "weight_of_ones": get_weight_of_ones(train_data["y"]),
    }

    model = CNN_Model(parameters)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    trainer.save_checkpoint(args.output_path)
