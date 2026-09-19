# infrastructure/persistence.py
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def checkpoint_saver(
    db_path: str = "data/checkpoints.sqlite",
) -> Iterator[SqliteSaver]:
    path = Path(db_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SqliteSaver.from_conn_string(
        str(path)
    ) as saver:
        yield saver


def build_checkpoint_saver(
    db_path: str = "data/checkpoints.sqlite",
):
    """
    Return a persistent saver whose connection remains open.

    The caller is responsible for closing it at shutdown.
    """
    path = Path(db_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return SqliteSaver.from_conn_string(
        str(path)
    )