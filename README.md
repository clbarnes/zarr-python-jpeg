# zarr-ext-jpeg

> Warning: the jpeg codec is not yet stable and may change.

The [`jpeg` extension codec](https://github.com/zarr-developers/zarr-extensions/pull/66) for [zarr-python](https://pypi.org/project/zarr/).

The JPEG codec is an array-to-bytes codec (aka "serializer") for compressing 2D data with 1 or 3 channels, where the channel axis must come last.
Use the [`reshape`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/reshape) and/or [`transpose`](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html) codecs to rearrange any arrays which do not fit this pattern.

You may also be interested in the [`template`](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/template) chunk key encoding,
to give all chunks the `.jpeg` extension so that they can be previewed by extension-sensitive browsers.

## Usage

Simply installing `zarr-ext-jpeg` (e.g. with pip or uv) will allow you to decode JPEG-encoded chunks when using zarr-python.

To create JPEG-encoded arrays, use the builder pattern to ensure that your codec is valid:

```python
from zarr.storage import MemoryStore
from zarr_ext_jpeg import JpegCodec
import numpy as np

my_codec = (
    JpegCodec
    # encoded color space:
    # determines which other parameters are valid
    .builder("ycbcr")
    # 2 horizontal and 2 vertical luminance pixels
    # per chrominance pixel
    .chroma_subsampling(2, 2)
    # defaults to 95 if not given
    .quality(75)
    .build()
)

store = MemoryStore()
rng = np.random.default_rng(1991)
data = rng.randint(0, 256, (10, 20), "uint8")
my_array = zarr.create_array(
    store,
    data=data,
    serializer=my_codec,
    # skip adding the default bytes-to-bytes codec(s)
    compressors=[],
)
```

## Lossy encoding warning

JPEG encoding is lossy.
Decoding and then re-encoding the same chunks will introduce more and more artifacts.
The `jpeg` codec should only be used for write-once data when whole chunks can be written at once.

## Parameter selection

### Quality

The default is 95, on a scale of 0-100.
Values above 95 lead to significantly larger files with very little perceptual gain.
Lower values introduce more artifacts.

### Channels

If your data doesn't have exactly 1 or 3 channels,
consider using the chunk configuration or sharding to make the chunk's channel axis length 1,
and then using grayscale encoding.

If you have 3 channels but they are not the red/green/blue (or YCbCr) of a natural scene,
use RGB->RGB encoding.
The default RGB->YCbCr color transform may introduce additional artifacts.

### Chunk/ shard shape

JPEGs are encoded in Minimum Coded Units whose shape depends on the chroma subsampling used.
MCUs are 8x8 pixels, multiplied by the chroma subsampling ratio: e.g. for a subsampling ratio of `[4, 1]`, the MCU would have shape `(32, 8)`.
The first two elements of your chunk shape SHOULD be a multiple of this MCU shape.

### Additional compression

At time of writing, by default `zarr-python` applies an additional bytes-to-bytes codec ("compressor") if none is specified.
As JPEG compresses quite effectively, this will likely not reduce your data size much and cost processing time;
explicitly set `compressors=[]` on array creation.

## Difference between `jpeg` and `imagecodecs_jpeg` codecs

`jpeg` is a registered zarr extension with implementations in multiple languages, targeting a minimal common API surface.
`imagecodecs_jpeg` supports more features, but is specified only by the details of a single implementation.

## Difference between `zarr-ext-jpeg` and `zarr-jpeg` packages

[zarr-jpeg](https://github.com/d-v-b/zarr-jpeg/) is a Zarr v2-era package which predates the current `jpeg` extension codec specification,
and indeed the Zarr v3 extension concept.
It is non-standard and defined only by the one implementation.
