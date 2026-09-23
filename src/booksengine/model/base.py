"""Общий протокол рекомендателя: обучение на толпе, выдача новому человеку по его оценкам (fold-in)."""
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from booksengine.model.matrix import RatingMatrix


class Recommender(Protocol):
    name: str

    def fit(self, train: RatingMatrix) -> None: ...

    def configure(self, **score_params) -> None:
        """Настройки выдачи, не требующие переобучения (число соседей, урезание весов)."""

    def score(self, inputs: sp.csr_matrix) -> np.ndarray:
        """Оценки новых людей (строки × все произведения) → баллы, float32. Только сохранённые артефакты + вход."""

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Recommender": ...


def write_params(path: Path, params: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1))


def read_params(path: Path) -> dict:
    return json.loads((path / "params.json").read_text())


def fingerprint(path: Path) -> str:
    """Отпечаток сохранённой модели: params.json и все .npy/.npz папки (не вложенные)."""
    h = hashlib.blake2b(digest_size=16)
    for f in sorted([*path.glob("*.np[yz]"), path / "params.json"]):
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()
