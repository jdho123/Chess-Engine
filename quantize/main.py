# pylint: disable=no-member
import tensorflow as tf
import numpy as np
import argparse
import yaml

# pylint: disable=unused-import
from train.model import SparseAccumulator, PerspectiveMerge, BucketedDense
from quantize.export_header import export_to_header
from quantize.quantize import quantize


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    QA = config["QA"]
    QB = config["QB"]

    model = tf.keras.models.load_model(config["model_path"])

    sparse_acc = next(l for l in model.layers if isinstance(l, SparseAccumulator))
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    bucketed = next(l for l in model.layers if isinstance(l, BucketedDense))
    hidden_dense = dense_layers[0]
    output_dense = dense_layers[1]

    acc_w = sparse_acc.embedding.get_weights()[0]
    acc_b = sparse_acc.bias.numpy()

    hidden_w, hidden_b = hidden_dense.get_weights()

    bucket_w = bucketed.kernel.numpy()
    bucket_b = bucketed.bias.numpy()

    out_w, out_b = output_dense.get_weights()

    arrays_dict = {
        "accumulator_weights": quantize(acc_w.transpose(), QA, np.int16),
        "accumulator_bias": quantize(acc_b, QA, np.int16),
        "hidden_weights": quantize(hidden_w.transpose(), QB, np.int8),
        "hidden_bias": quantize(hidden_b, QA * QB, np.int32),
        "bucketed_weights": quantize(bucket_w.transpose(0, 2, 1), QB, np.int8),
        "bucketed_bias": quantize(bucket_b, QA * QB, np.int32),
        "output_weights": quantize(out_w.transpose(), QB, np.int8),
        "output_bias": quantize(out_b, QA * QB, np.int32),
    }

    export_to_header(
        config["header_output_path"],
        config["cpp_output_path"],
        config["header_template_path"],
        config["cpp_template_path"],
        arrays_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a chess evaluation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="model/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    main(args.config)
