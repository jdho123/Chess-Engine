import tensorflow as tf
import random
import glob
import os

COMPRESSION_EXT = {"GZIP": ".tfrecord.gz", "ZLIB": ".tfrecord.zlib", "": ".tfrecord"}

FEATURE_DESCRIPTION = {
    "white": tf.io.FixedLenFeature([], tf.string),
    "black": tf.io.FixedLenFeature([], tf.string),
    "material_value": tf.io.FixedLenFeature([], tf.string),
    "turn": tf.io.FixedLenFeature([], tf.string),
    "score": tf.io.FixedLenFeature([], tf.string),
}


def parse_tfrecord_fn(example_protos: tf.Tensor):
    parsed = tf.io.parse_example(example_protos, FEATURE_DESCRIPTION)

    white = tf.io.decode_raw(parsed["white"], tf.uint16)
    black = tf.io.decode_raw(parsed["black"], tf.uint16)
    material_value = tf.io.decode_raw(parsed["material_value"], tf.uint16)
    turn = tf.io.decode_raw(parsed["turn"], tf.uint8)
    score = tf.io.decode_raw(parsed["score"], tf.float32)

    white = tf.reshape(white, (-1, 32))
    black = tf.reshape(black, (-1, 32))
    material_value = tf.reshape(material_value, (-1, 1))
    turn = tf.reshape(turn, (-1, 1))
    score = tf.reshape(score, (-1, 1))

    return {
        "white": white,
        "black": black,
        "material_value": material_value,
        "turn": turn,
    }, score


def create_dataset(
    data_dir: str,
    batch_size: int,
    test_split: float,
    block_length: int,
    shuffle_buffer: int,
    seed: int = 42,
    compression_type: str = "GZIP",
) -> tf.data.Dataset:
    ext = COMPRESSION_EXT[compression_type]
    file_pattern = os.path.join(data_dir, f"*{ext}")
    shards = sorted(glob.glob(file_pattern))

    if not shards:
        raise FileNotFoundError(
            f"No TFRecord files found in '{data_dir}' matching pattern '*{ext}'"
        )

    random.seed(seed)
    random.shuffle(shards)

    num_files = len(shards)
    num_test_files = int(num_files * test_split)

    if num_test_files == 0 and num_files > 1:
        num_test_files = 1

    options = tf.io.TFRecordOptions(compression_type=compression_type)
    approx_shard_samples = sum(
        1 for _ in tf.compat.v1.io.tf_record_iterator(shards[0], options=options)
    )
    approx_train_samples = approx_shard_samples * (num_files - num_test_files)
    approx_test_samples = approx_shard_samples * num_test_files

    test_files = shards[:num_test_files]
    train_files = shards[num_test_files:]

    def _build_pipeline(
        file_list: list[str], is_training: bool = True
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(file_list)

        if is_training:
            dataset = dataset.shuffle(len(file_list), seed=seed)

        dataset = dataset.interleave(
            lambda f: tf.data.TFRecordDataset(f, compression_type=compression_type),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            block_length=block_length,
            deterministic=not is_training,
        )

        if is_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size, drop_remainder=True)

        dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    train_dataset = _build_pipeline(train_files, is_training=True)
    test_dataset = _build_pipeline(test_files, is_training=False)

    return (approx_train_samples, train_dataset), (approx_test_samples, test_dataset)
