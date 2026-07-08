# pylint: disable=no-member
import tensorflow as tf
import argparse
import logging
import yaml
import sys
import os

from train.dataset import create_dataset
from train.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tf.keras.utils.set_random_seed(config["seed"])

    batch_size = config["batch_size"]
    (train_samples, train_dataset), (val_samples, val_dataset) = create_dataset(
        config["data_dir"],
        batch_size,
        config["test_split"],
        config["block_length"],
        config["shuffle_buffer"],
        config["seed"],
    )

    model = build_model(
        input_dim=config["input_dim"],
        embedding_dim=config["embedding_dim"],
        accumulator_dim=config["accumulator_dim"],
        hidden_dim=config["hidden_dim"],
        bucket_params={"num_buckets": 32},
    )
    model.summary()

    steps_per_train_pass = train_samples // batch_size
    steps_per_val_pass = val_samples // batch_size
    total_steps = steps_per_train_pass * config["epochs"]

    val_per_pass = 4

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - steps_per_train_pass,
        warmup_target=config["peak_lr"],
        warmup_steps=steps_per_train_pass,
        alpha=0.0,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        jit_compile=True,
    )

    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["model_dir"], "chess_eval.keras"),
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"]),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config["epochs"] * val_per_pass,
        steps_per_epoch=steps_per_train_pass // val_per_pass,
        validation_steps=steps_per_val_pass,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chess evaluation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="model/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    main(args.config)
