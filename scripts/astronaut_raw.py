#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["numpy", "pillow", "zarr"]
# ///
"""
Create a zarr array for the raw astronaut data.
"""

import shutil
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

with Image.open(DATA / "astronaut.png") as im:
    img: np.ndarray = np.asarray(im)

out_dir = DATA / "astronaut.zarr"
if out_dir.is_dir():
    shutil.rmtree(out_dir)

g = zarr.open_group(out_dir)
del g["raw"]
g.create_array("raw", data=img, chunks=img.shape)
