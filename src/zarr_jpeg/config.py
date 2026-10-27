from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Self, cast

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.common import JSON, MemoryOrder
from zarr.core.dtype import UInt8
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
from zarr.core.metadata.v3 import ChunkGridMetadata
from zarr.registry import register_codec

from .pillow import pillow_decode, pillow_encode

type ColorSpace = Literal["grayscale", "rgb", "ycbcr"]

COLORSPACES: dict[ColorSpace, ColorSpaceInfo] = {}
CODEC_NAME = "jpeg"


@dataclass(frozen=True)
class ColorSpaceInfo:
    name: ColorSpace
    n_components: int
    can_encode_to: set[ColorSpace]
    can_decode_to: set[ColorSpace]
    default_decodes_to: ColorSpace


def _register_colorspace(info: ColorSpaceInfo):
    COLORSPACES[info.name] = info


_register_colorspace(
    ColorSpaceInfo("grayscale", 1, {"grayscale"}, {"grayscale"}, "grayscale")
)
_register_colorspace(ColorSpaceInfo("rgb", 3, {"rgb", "ycbcr"}, {"rgb"}, "rgb"))
_register_colorspace(ColorSpaceInfo("ycbcr", 3, {"ycbcr"}, {"rgb", "ycbcr"}, "rgb"))


@dataclass(frozen=True)
class SamplingFactor:
    horizontal: int
    vertical: int

    def __iter__(self) -> Iterator[int]:
        yield self.horizontal
        yield self.vertical

    def __post_init__(self):
        for v in (self.horizontal, self.vertical):
            if v < 1 or v > 4:
                raise ValueError(
                    f"Chroma subsampling factor must be in [1, 4]; got {v}"
                )

    def check_trivial(self):
        if self.horizontal != 1 or self.vertical != 1:
            raise ValueError(f"Expected trivial sampling factor, got {self}")

    @classmethod
    def trivial(cls) -> SamplingFactor:
        return SamplingFactor(1, 1)


@dataclass(frozen=True)
class JpegCodec(ArrayBytesCodec):
    """Array-to-bytes codec for JPEG compression.

    It is recommended to use `JpegCodec.builder()` to construct the codec.
    """

    quality: int
    subsampling: tuple[SamplingFactor, ...]
    encoded_color_space: ColorSpace
    decoded_color_space: ColorSpace

    def to_dict(self) -> dict[str, JSON]:
        d = {
            "quality": self.quality,
            "subsampling": tuple(
                (sf.horizontal, sf.vertical) for sf in self.subsampling
            ),
            "encoded_color_space": self.encoded_color_space,
        }
        if (
            COLORSPACES[self.encoded_color_space].default_decodes_to
            != self.decoded_color_space
        ):
            d["decoded_color_space"] = self.decoded_color_space
        return {"name": CODEC_NAME, "configuration": d}

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        if data.get("name") != CODEC_NAME:
            raise ValueError(
                f"Expected codec name {CODEC_NAME}, got {data.get('name')}"
            )

        config = cast(dict[str, JSON], data.get("configuration", {}))

        ss = tuple(
            SamplingFactor(*sf) for sf in cast(list[list[int]], config["subsampling"])
        )
        enc_cs = cast(ColorSpace, config["encoded_color_space"])
        qual = cast(int, config["quality"])
        dec_cs = config.get("decoded_color_space")
        if dec_cs is None:
            dec_cs = COLORSPACES[enc_cs].default_decodes_to
        else:
            dec_cs = cast(ColorSpace, dec_cs)
        return cls(qual, ss, enc_cs, dec_cs)

    def __post_init__(self):
        if not isinstance(self.quality, int) or self.quality < 0 or self.quality > 100:
            raise ValueError(f"JPEG quality must be from 0 to 100; got {self.quality}")

        if (
            self.encoded_color_space
            not in COLORSPACES[self.decoded_color_space].can_encode_to
        ):
            raise ValueError(
                f"Cannot encode color space '{self.decoded_color_space}' to '{self.encoded_color_space}'"
            )

        if (
            self.decoded_color_space
            not in COLORSPACES[self.encoded_color_space].can_decode_to
        ):
            raise ValueError(
                f"Cannot decode color space '{self.encoded_color_space}' to '{self.decoded_color_space}'"
            )

        n_comp = COLORSPACES[self.encoded_color_space].n_components
        if len(self.subsampling) != n_comp:
            raise ValueError(
                f"JPEG subsampling must have same length as the encoded color space ('{self.encoded_color_space}')'s number of components ({n_comp}); got {len(self.subsampling)}"
            )

        for ss in self.subsampling[1:]:
            ss.check_trivial()

    def _validate_chunk(
        self, dtype: ZDType, shape: tuple[int, ...], order: MemoryOrder = "C"
    ):
        """Check that data type and shape are valid for this configuration"""
        if dtype != UInt8():
            raise ValueError("JPEG can only encode uint8 data")
        u16max = 2**16 - 1

        n_comp = COLORSPACES[self.decoded_color_space].n_components
        match len(shape):
            case 2:
                if n_comp != 1:
                    raise ValueError(f"JPEG must have {n_comp} channels; got 1")
            case 3:
                if shape[2] != n_comp:
                    raise ValueError(
                        f"JPEG must have {n_comp} channels; got {shape[2]}"
                    )
            case n:
                raise ValueError(f"JPEG data must be 2 or 3D; got {n}D")

        if shape[0] > u16max or shape[1] > u16max:
            raise ValueError(
                f"JPEGs must be smaller than 2^16px per side, got ({shape[:2]})"
            )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        """Given an input byte length, this method returns the output byte length.
        Raises a NotImplementedError for codecs with variable-sized outputs (e.g. compressors).

        Parameters
        ----------
        input_byte_length : int
        chunk_spec : ArraySpec

        Returns
        -------
        int
        """
        raise NotImplementedError

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        """Validates that the codec configuration is compatible with the array metadata.
        Raises errors when the codec configuration is not compatible.

        Parameters
        ----------
        shape : tuple[int, ...]
            The array shape
        dtype : np.dtype[Any]
            The array data type
        chunk_grid : ChunkGridMetadata
            The array chunk grid metadata
        """
        if dtype != UInt8():
            raise ValueError(f"JPEG codec is only defined for uint8 data, got {dtype}")

    def _decode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        self._validate_chunk(chunk_spec.dtype, chunk_spec.shape, chunk_spec.order)
        arr = pillow_decode(self, chunk_data.as_buffer_like(), chunk_spec.shape)
        if arr.ndim == 2 and chunk_spec.ndim == 3:
            arr = np.expand_dims(arr, -1)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(arr)

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(self._decode_sync, chunk_data, chunk_spec)

    def _encode_sync(self, chunk_data: NDBuffer, chunk_spec: ArraySpec):
        self._validate_chunk(chunk_spec.dtype, chunk_spec.shape, chunk_spec.order)
        # TODO: get rid of this possible copy?
        b = pillow_encode(self, chunk_data.as_numpy_array())
        return chunk_spec.prototype.buffer.from_bytes(b)

    async def _encode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_data, chunk_spec)

    @classmethod
    def builder(cls, encoded_color_space: ColorSpace) -> JpegCodecBuilder:
        return JpegCodecBuilder(encoded_color_space)


@dataclass(init=False)
class JpegCodecBuilder:
    _encoded_color_space: ColorSpace
    _decoded_color_space: ColorSpace
    _subsampling_first: SamplingFactor
    _quality: int

    def __init__(self, encoded_color_space: ColorSpace):
        self._encoded_color_space = encoded_color_space
        self._decoded_color_space = COLORSPACES[encoded_color_space].default_decodes_to
        self._subsampling_first = SamplingFactor.trivial()
        self._quality = 95

    def encode_from_ycbcr(self) -> Self:
        """Decoded pixels are in interleaved YCbCr format.

        Defaults to `False` (expects interleaved RGB pixels).
        Errors unless encoded_color_space is `"ycbcr"`."""
        if self._encoded_color_space != "ycbcr":
            raise ValueError("Can only encode ycbcr from ycbcr")
        self._decoded_color_space = "ycbcr"
        return self

    def chroma_subsampling(self, horizontal: int, vertical: int) -> Self:
        """Number of horizontal and vertical luminance pixels per chrominance pixel.

        Defaults to `(1, 1)` (no subsampling).
        Errors unless encoded_color_space is `"ycbcr"`.
        """
        if self._encoded_color_space != "ycbcr":
            raise ValueError("Can only subsamble ycbcr-encoded data")
        self._subsampling_first = SamplingFactor(horizontal, vertical)
        return self

    def quality(self, quality: int) -> Self:
        """Quality in `[0, 100]`.

        Defaults to 95.
        """
        if quality < 0 or quality > 100:
            raise ValueError("quality must be in [0, 100]")
        self._quality = quality
        return self

    def build(self) -> JpegCodec:
        """Build the codec."""
        trivial_subsamps = COLORSPACES[self._encoded_color_space].n_components - 1
        subsamps = [self._subsampling_first] + [
            SamplingFactor.trivial()
        ] * trivial_subsamps

        return JpegCodec(
            self._quality,
            tuple(subsamps),
            self._encoded_color_space,
            self._decoded_color_space,
        )


register_codec(
    "jpeg",
    JpegCodec,
)
