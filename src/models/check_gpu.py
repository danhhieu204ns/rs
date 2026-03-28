from __future__ import annotations

import json

import tensorflow as tf


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    info = {
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [str(g) for g in gpus],
        "cpus": [str(c) for c in cpus],
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
