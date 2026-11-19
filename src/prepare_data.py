"""Prépare le jeu de données : contrôle qualité + découpage train / val / test reproductible.

Exemple :
    python prepare_data.py --src photos_brutes --out data_split --split 80 10 10 --seed 42

Produit :
    data_split/train/, data_split/val/, data_split/test/   (copies ; --link pour des liens durs)
    data_split/splits.json                                  (liste des fichiers de chaque ensemble)
    data_split/dataset_report.json                          (statistiques utiles pour le rapport)

Règles :
  - le découpage est fait UNE fois, avec une graine fixe : tout le monde obtient les mêmes ensembles ;
  - le jeu de test n'est utilisé que par evaluate.py, jamais pendant l'entraînement ;
  - les doublons exacts (même contenu) sont écartés AVANT le découpage pour éviter les fuites train/test.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="dossier contenant les images (sous-dossiers acceptés)")
    p.add_argument("--out", default="data_split")
    p.add_argument("--split", type=float, nargs=3, default=[80, 10, 10], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-size", type=int, default=128, help="côté minimal en pixels (images plus petites écartées)")
    p.add_argument("--min-chroma", type=float, default=3.0,
                   help="chroma RVB moyenne minimale : écarte les images déjà en noir et blanc (inutiles pour apprendre la couleur)")
    p.add_argument("--max-images", type=int, default=0, help="limiter le nombre d'images (0 = toutes)")
    p.add_argument("--link", action="store_true", help="liens durs au lieu de copies (même disque uniquement)")
    return p.parse_args()


def rgb_chroma(im):
    """Écart moyen entre canaux R, G, B (0 = image en niveaux de gris) sur une miniature."""
    t = im.copy()
    t.thumbnail((64, 64))
    a = np.asarray(t, dtype=np.int16)
    return float((a.max(-1) - a.min(-1)).mean())


def main():
    a = parse_args()
    if abs(sum(a.split) - 100) > 1e-6:
        raise SystemExit("--split doit totaliser 100 (ex. 80 10 10)")
    src, out = Path(a.src), Path(a.out)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXT and p.is_file())
    if not files:
        raise SystemExit(f"Aucune image trouvée dans {src}")

    stats = {"found": len(files), "unreadable": 0, "too_small": 0, "grayscale": 0, "duplicates": 0}
    seen, kept = set(), []
    for p in files:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if min(im.size) < a.min_size:
                    stats["too_small"] += 1
                    continue
                if rgb_chroma(im) < a.min_chroma:
                    stats["grayscale"] += 1
                    continue
        except (UnidentifiedImageError, OSError):
            stats["unreadable"] += 1
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            stats["duplicates"] += 1
            continue
        seen.add(h)
        kept.append(p)

    rnd = random.Random(a.seed)
    rnd.shuffle(kept)
    if a.max_images:
        kept = kept[: a.max_images]
    n = len(kept)
    n_tr, n_va = int(n * a.split[0] / 100), int(n * a.split[1] / 100)
    parts = {"train": kept[:n_tr], "val": kept[n_tr:n_tr + n_va], "test": kept[n_tr + n_va:]}
    if min(len(v) for v in parts.values()) == 0:
        raise SystemExit(f"Pas assez d'images ({n}) pour ce découpage : {({k: len(v) for k, v in parts.items()})}")

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} existe déjà et n'est pas vide : supprimez-le pour régénérer le découpage.")
    index = {}
    for name, items in parts.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        index[name] = []
        for i, p in enumerate(items):
            dst = d / f"{i:06d}{p.suffix.lower()}"      # noms uniques : évite les collisions entre sous-dossiers
            if a.link:
                os.link(p, dst)
            else:
                shutil.copy2(p, dst)
            index[name].append({"file": dst.name, "source": str(p.relative_to(src))})

    report = {"seed": a.seed, "split_percent": a.split, "kept": n,
              "counts": {k: len(v) for k, v in parts.items()}, "filters": {
                  "min_size": a.min_size, "min_chroma": a.min_chroma}, "discarded": stats}
    (out / "splits.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nPrêt : {out}/train, {out}/val, {out}/test")


if __name__ == "__main__":
    main()
