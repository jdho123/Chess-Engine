from collections.abc import Generator
import multiprocessing as mp
import zstandard as zstd
import tensorflow as tf
from tqdm import tqdm
import argparse
import hashlib
import sys
import os
import io

from preprocess.feature_extract import ChessFeatureExtractor
from preprocess.serialize import serialize_to_tfrecord


def _process_batch(
    lines: list[bytes],
    feature_extractor: ChessFeatureExtractor,
    writer: tf.io.TFRecordWriter,
):
    for line in lines:
        result = feature_extractor.process_line(line)
        if result is None:
            continue

        w, b, t, s = result

        record_bytes = serialize_to_tfrecord(w, b, t, s)
        writer.write(record_bytes)


def _batch_generator(
    reader: zstd.ZstdDecompressionReader, batch_size: int
) -> Generator:
    current_batch = []

    for line in reader:
        line = line.strip()
        if not line:
            continue

        current_batch.append(line)

        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch


def _tfrecord_creation_worker(
    worker_id: int,
    file_idx: int,
    input_queue: mp.Queue,
    output_dir: str,
    target_mb: int,
    check_interval: int = 10_000,
):
    target_bytes = target_mb * 1024 * 1024

    def get_filepath(worker_id: int, file_idx: int) -> str:
        file_id_bytes = f"{worker_id}_{file_idx}".encode("utf-8")
        filename = f"shard-{hashlib.sha256(file_id_bytes).hexdigest()}.tfrecord.gz"

        return os.path.join(output_dir, filename)

    current_file = get_filepath(worker_id, file_idx)

    feature_extractor = ChessFeatureExtractor()
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    writer = tf.io.TFRecordWriter(current_file, options=options)

    records_since_check = 0

    while True:
        batch = input_queue.get()

        if batch is None:
            writer.close()

            sys.exit(99)

        _process_batch(batch, feature_extractor, writer)
        records_since_check += len(batch)

        if records_since_check >= check_interval:
            writer.flush()

            if os.path.getsize(current_file) >= target_bytes:
                writer.close()
                sys.exit(0)

            records_since_check = 0


def _worker_supervisor(
    worker_id: int, input_queue: str, output_dir: str, target_mb: int
):
    file_idx = 0

    while True:
        p = mp.Process(
            target=_tfrecord_creation_worker,
            args=(worker_id, file_idx, input_queue, output_dir, target_mb),
        )
        p.start()
        p.join()

        if p.exitcode == 0:
            file_idx += 1
        elif p.exitcode == 99:
            break


def main(args: dict):
    os.makedirs(args["output_dir"], exist_ok=True)

    dctx = zstd.ZstdDecompressor()

    queue = mp.Queue(maxsize=args["num_workers"] * 4)
    workers = []
    for i in range(args["num_workers"]):
        p = mp.Process(
            target=_worker_supervisor,
            args=(i, queue, args["output_dir"], args["target_mb"]),
        )
        p.start()
        workers.append(p)

    with (
        open(args["input_file"], "rb") as f,
        dctx.stream_reader(f) as reader,
    ):
        buffered_stream = io.BufferedReader(reader)
        for batch in tqdm(_batch_generator(buffered_stream, args["batch_size"])):
            queue.put(batch)

    for _ in range(len(workers)):
        queue.put(None)  # Poison pill

    for p in workers:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallelized conversoin of Zstd-compressed jsonl chess position data to GZIP-compressed TFRecords."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input compressed .zst file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the generated .tfrecord shards will be saved.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel worker processes. Defaults to CPU count - 1.",
    )

    parser.add_argument(
        "--target_mb",
        type=int,
        default=100,
        help="Target size for each .tfrecord shard in megabytes (MB). Default is 150.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of lines sent to a worker in a single IPC queue batch. Default is 1024.",
    )

    args_dict = vars(parser.parse_args())

    main(args_dict)
