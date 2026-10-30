import re
from collections.abc import Sequence
from pathlib import Path
from types import CodeType
from typing import cast

import numpy as np
import pytest
import zarr
from zarr_template import TemplateChunkKeyEncoding

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


@pytest.fixture
def astronaut() -> np.ndarray:
    arr = zarr.open_array(DATA_DIR / "astronaut.zarr" / "raw", mode="r")
    return cast(np.ndarray, arr[:])


@pytest.fixture
def chunk_key_encoding():
    return TemplateChunkKeyEncoding("c/{*}.jpeg", "_")


@pytest.fixture(scope="module")
def readme_blocks() -> Sequence[CodeType]:
    path = PROJECT_DIR / "README.md"
    txt = path.read_text()
    blocks = []
    for m in re.finditer(r"^```python\n(.*?)^```$", txt, re.MULTILINE | re.DOTALL):
        # pad the code block with leading newlines
        # so that line counts point to the right place in stack traces
        leading_lines = txt.count("\n", 0, m.start(1))
        code = "\n" * leading_lines + m.group(1)
        compiled = compile(code, str(path), "exec")
        blocks.append(compiled)
    return tuple(blocks)
