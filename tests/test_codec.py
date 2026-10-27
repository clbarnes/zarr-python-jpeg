import io
from typing import cast

import numpy as np
import zarr
from PIL import Image
from zarr.storage import MemoryStore
from zarr_template import TemplateChunkKeyEncoding

from zarr_ext_jpeg import JpegCodec
from zarr_ext_jpeg.util import rgb_to_ycbcr


def check_roundtrip(
    arr: np.ndarray, jpeg: JpegCodec, chunks: tuple[int, ...] | None = None
):
    if chunks is None:
        chunks = arr.shape
    store = MemoryStore()
    za = zarr.create_array(
        store,
        data=arr,
        serializer=jpeg,
        chunks=chunks,
        chunk_key_encoding=TemplateChunkKeyEncoding("c/{*}.jpeg", "_"),
        compressors=[],
    )

    # check that all the chunks are valid JFIFs
    for k, v in store._store_dict.items():
        if k.endswith("zarr.json"):
            continue
        b = v.to_bytes()
        with Image.open(io.BytesIO(b), formats=["JPEG"]) as im:
            im.verify()

    arr2 = cast(np.ndarray, za[:])
    assert arr2.shape == arr.shape


def test_roundtrip_rgb(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec.builder("rgb").build(),
        (256, 256, 3),
    )


def test_roundtrip_rgb_ycbcr(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec.builder("ycbcr").chroma_subsampling(2, 2).build(),
        (256, 256, 3),
    )


def test_roundtrip_ycbcr(astronaut: np.ndarray):
    ycbcr = rgb_to_ycbcr(astronaut)
    check_roundtrip(
        ycbcr,
        JpegCodec.builder("ycbcr").encode_from_ycbcr().chroma_subsampling(2, 2).build(),
    )


def test_roundtrip_grayscale(astronaut: np.ndarray):
    check_roundtrip(
        astronaut[:, :, 0],
        JpegCodec.builder("grayscale").build(),
        (256, 256),
    )


def test_roundtrip_chunk_to_grayscale(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec.builder("grayscale").build(),
        (256, 256, 1),
    )
