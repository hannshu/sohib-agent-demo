from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "clean" / "knowledge_base.json"


@lru_cache(maxsize=1)
def load(path: Path = _DEFAULT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def save(kb: dict, path: Path = _DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(kb, f, indent=2)
    load.cache_clear()
