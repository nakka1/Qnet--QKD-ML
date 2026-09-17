"""Dataset provenance: what was generated, from what, with which seed.

A synthetic dataset without its generating parameters is close to worthless six
months later — you cannot tell whether a model got better or the data got
easier.  Worse, in this domain the parameters *are* the experiment: a
polarisation dataset with a 10-minute thermal correlation time and one with a
10-second correlation time look identical in a plot and pose completely
different learning problems.

So every generator in this package returns a :class:`Dataset`: a DataFrame plus
the full parameter set, the seed, the software versions, and a one-line
description of what the rows mean.  :meth:`Dataset.save` writes both, side by
side, and :func:`load` brings them back together.

Format is CSV plus a JSON sidecar by default — readable by anything, diffable,
and with no dependency beyond pandas.  Parquet is used instead when a Parquet
engine is installed and the caller asks for it.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["Dataset", "DatasetMetadata", "load", "describe_columns"]


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        from ..compat import bootstrap

        status = bootstrap()
        versions["sequence"] = status.sequence_version or "unknown"
        if status.shimmed:
            versions["sequence_shimmed_deps"] = ",".join(sorted(status.shimmed))
    except Exception:  # pragma: no cover - provenance must never break generation
        versions["sequence"] = "unavailable"
    return versions


@dataclass
class DatasetMetadata:
    """Everything needed to regenerate a dataset byte-for-byte.

    Attributes:
        name: short identifier, e.g. ``polarization_drift``.
        description: one line on what a row represents.
        parameters: the generator's arguments, JSON-serialisable.
        seed: the RNG seed used.
        columns: column name to one-line meaning.
        created_utc: ISO timestamp.
        software: versions of the packages involved, including which SeQUeNCe
            dependencies were shimmed (a dataset generated under the
            compatibility shims should say so).
        row_count: number of rows.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    columns: dict[str, str] = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software: dict[str, str] = field(default_factory=_software_versions)
    row_count: int = 0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=_json_default, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "DatasetMetadata":
        return cls(**json.loads(text))


def _json_default(value):
    """Make numpy scalars, arrays and paths JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


@dataclass
class Dataset:
    """A DataFrame that remembers where it came from."""

    frame: pd.DataFrame
    metadata: DatasetMetadata

    def __post_init__(self) -> None:
        self.metadata.row_count = int(len(self.frame))
        undocumented = set(self.frame.columns) - set(self.metadata.columns)
        if undocumented:
            raise ValueError(
                f"dataset {self.metadata.name!r} has undocumented columns: "
                f"{sorted(undocumented)} — add them to metadata.columns")

    def __len__(self) -> int:
        return len(self.frame)

    def save(self, path: str | Path, format: str = "csv") -> Path:
        """Write the data and its metadata sidecar.

        Args:
            path: destination for the data.  The sidecar is written next to it
                with a ``.meta.json`` suffix.
            format: ``"csv"`` or ``"parquet"``.  Parquet needs a Parquet engine
                (pyarrow or fastparquet) installed.

        Returns:
            The path actually written.

        Raises:
            ValueError: on an unknown format.
            ImportError: if Parquet was asked for and no engine is available.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            self.frame.to_csv(path, index=False)
        elif format == "parquet":
            self.frame.to_parquet(path, index=False)
        else:
            raise ValueError(f"unknown format {format!r}; expected 'csv' or 'parquet'")

        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.write_text(self.metadata.to_json())
        return path

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.frame.head(n)

    def describe(self) -> str:
        """A human-readable summary: what this is, how it was made, what is in it."""
        lines = [f"{self.metadata.name}: {self.metadata.description}",
                 f"  {len(self.frame)} rows x {len(self.frame.columns)} columns"
                 f"  (seed={self.metadata.seed})",
                 "  columns:"]
        for column, meaning in self.metadata.columns.items():
            lines.append(f"    {column:<28} {meaning}")
        return "\n".join(lines)


def load(path: str | Path) -> Dataset:
    """Read back a dataset and its metadata sidecar.

    Raises:
        FileNotFoundError: if the data file or its sidecar is missing.
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"{sidecar} is missing — a dataset without its metadata cannot be trusted")

    frame = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    return Dataset(frame=frame, metadata=DatasetMetadata.from_json(sidecar.read_text()))


def describe_columns(columns: Mapping[str, str], frame: pd.DataFrame) -> dict[str, str]:
    """Keep only the documented columns that are actually present.

    Generators build their column dictionary once and then emit different
    subsets depending on options; this trims the documentation to match.
    """
    return {name: meaning for name, meaning in columns.items() if name in frame.columns}
