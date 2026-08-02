from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOUG_DESCRIPTION = (
    "SOUG n=50, n_set=5n, set sizes sampled uniformly from {2,3,4,5,6}, "
    "grand worth 100."
)
WINE_FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


def stable_seed(*parts: object) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def csv_bytes(frame: pd.DataFrame, lineterminator: str = "\n") -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator=lineterminator)
    return buffer.getvalue().encode("utf-8")


def write_bytes(path: Path, content: bytes, overwrite: bool) -> str:
    if path.exists():
        if path.read_bytes() == content:
            return "verified"
        if not overwrite:
            raise FileExistsError(f"Existing file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return "wrote"


def write_soug(path: Path, overwrite: bool) -> str:
    n = 50
    n_set = 5 * n
    base_seed = 20260519
    rep = 0
    rng = np.random.default_rng(stable_seed(base_seed, "size_2_6", n, rep))
    set_sizes = rng.integers(2, 7, size=n_set).astype(np.int16)
    sets = np.full((n_set, 6), -1, dtype=np.int32)
    for row, size in enumerate(set_sizes):
        sets[row, :size] = np.sort(rng.choice(n, size=int(size), replace=False))
    coefficients = rng.random(n_set)
    coefficients = coefficients / coefficients.sum() * 100.0
    coefficients[-1] += 100.0 - float(coefficients.sum())

    expected = {
        "sets": sets,
        "set_sizes": set_sizes,
        "coeffs": coefficients,
        "n_player": np.asarray(n, dtype=np.int64),
        "n_set": np.asarray(n_set, dtype=np.int64),
        "gw": np.asarray(100.0, dtype=np.float64),
        "seed": np.asarray(base_seed, dtype=np.int64),
        "rep": np.asarray(rep, dtype=np.int64),
        "description": np.asarray(SOUG_DESCRIPTION),
        "format": np.asarray("numeric_sets_v1"),
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as current:
            matches = set(current.files) == set(expected) and all(
                np.array_equal(current[key], value) for key, value in expected.items()
            )
        if matches:
            return "verified"
        if not overwrite:
            raise FileExistsError(f"Existing file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **expected)
    return "wrote"


def breast_cancer_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["id", *[f"f{i}" for i in range(1, 10)], "Y"]
    raw = pd.read_csv(
        ROOT / "data" / "original" / "breast-cancer-wisconsin.data",
        names=columns,
        dtype=str,
    )
    split = json.loads(
        (ROOT / "data" / "breast_cancer_n50" / "split.json").read_text()
    )
    raw["f6"] = (
        pd.to_numeric(raw["f6"], errors="coerce")
        .fillna(int(split["missing_bare_nuclei_value"]))
        .astype(int)
    )
    feature_columns = [f"f{i}" for i in range(1, 10)]
    for column in [*feature_columns[:5], *feature_columns[6:], "Y"]:
        raw[column] = pd.to_numeric(raw[column]).astype(int)
    output_columns = [*feature_columns, "Y"]
    players = raw.iloc[split["player_indices"]][output_columns].reset_index(drop=True)
    validation = raw.iloc[split["validation_indices"]][output_columns].reset_index(drop=True)
    return players, validation


def wine_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        ROOT / "data" / "original" / "wine.data",
        names=["class", *WINE_FEATURES],
        dtype=float,
    )
    x = raw[WINE_FEATURES].to_numpy(dtype=float)
    y = raw["class"].to_numpy(dtype=int) - 1
    random_state = stable_seed(20260522, "wine_split", 50) % (2**32)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        train_size=50,
        test_size=50,
        stratify=y,
        random_state=random_state,
    )
    players = pd.DataFrame(x_train, columns=WINE_FEATURES)
    players["Y"] = y_train
    validation = pd.DataFrame(x_valid, columns=WINE_FEATURES)
    validation["Y"] = y_valid
    return players, validation


def generate(dataset: str, output_root: Path, overwrite: bool) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    if dataset in {"all", "soug"}:
        path = output_root / "soug_n50" / "soug_n50.npz"
        records.append((write_soug(path, overwrite), path))
    if dataset in {"all", "breast_cancer"}:
        players, validation = breast_cancer_frames()
        for frame, path in [
            (players, output_root / "breast_cancer_n50" / "players.csv"),
            (validation, output_root / "breast_cancer_n50" / "validation.csv"),
        ]:
            lineterminator = "\r\n" if path.name == "validation.csv" else "\n"
            records.append(
                (write_bytes(path, csv_bytes(frame, lineterminator), overwrite), path)
            )
    if dataset in {"all", "wine"}:
        players, validation = wine_frames()
        for frame, path in [
            (players, output_root / "wine_n50" / "players.csv"),
            (validation, output_root / "wine_n50" / "validation.csv"),
        ]:
            records.append((write_bytes(path, csv_bytes(frame), overwrite), path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the bundled n=50 datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "soug", "breast_cancer", "wine"],
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for status, path in generate(args.dataset, args.output_root, args.overwrite):
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
