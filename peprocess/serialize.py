import tensorflow as tf
import numpy as np


def serialize_to_tfrecord(
    white: np.ndarray, black: np.ndarray, turn: np.ndarray, score: np.ndarray
) -> bytes:
    """Serializes chess feature arrays into a TFRecord Example.

    Encodes each input array as raw bytes within a tf.train.Example, ready
    to be written to a TFRecord file for TensorFlow data pipelines.

    Args:
        white: uint16[32] array of HalfKP feature indices from white
            king's perspective.
        black: uint16[32] array of HalfKP feature indices from black
            king's perspective.
        turn: uint8[1] array, 1 if white to move else 0.
        score: float32[1] array holding the win-probability target.

    Returns:
        Serialized bytes of a tf.train.Example protocol buffer, with
        "white", "black", "turn", and "score" byte-string features
        holding the raw bytes of each corresponding input array.
    """

    feature = {
        "white": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[white.tobytes()])
        ),
        "black": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[black.tobytes()])
        ),
        "turn": tf.train.Feature(bytes_list=tf.train.BytesList(value=[turn.tobytes()])),
        "score": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[score.tobytes()])
        ),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
