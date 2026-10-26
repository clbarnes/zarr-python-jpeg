from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zarr
from zarr_template import TemplateChunkKeyEncoding

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


@pytest.fixture
def astronaut() -> np.ndarray:
    arr = zarr.open_array(DATA_DIR / "astronaut.zarr" / "raw", mode="r")
    return cast(np.ndarray, arr[:])


@pytest.fixture
def chunk_key_encoding():
    return TemplateChunkKeyEncoding("c/{*}.jpeg", "_")
