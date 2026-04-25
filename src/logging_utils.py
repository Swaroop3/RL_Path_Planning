"""Logging helpers for training and simulation."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def configure_logging(log_dir: str | Path, name: str = "train") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(torch_threads: int | None = None, cudnn_benchmark: bool = True) -> None:
    if torch_threads:
        torch.set_num_threads(torch_threads)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark


class CsvMetricLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        clean = {field: row.get(field, "") for field in self.fieldnames}
        self._writer.writerow(clean)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")

    def write(self, row: dict[str, Any]) -> None:
        self._file.write(json.dumps(row, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp_path.replace(path)

