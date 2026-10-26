# zarr-jpeg

> Warning: the jpeg codec is not yet stable and may change.

The [`jpeg` extension codec](https://github.com/zarr-developers/zarr-extensions/pull/66) for [zarr-python](https://pypi.org/project/zarr/).

Currently, this is implemented as a thin wrapper over the [`imagecodecs_jpeg` codec](https://github.com/cgohlke/imagecodecs).

## Difference between `jpeg` and `imagecodecs_jpeg`

`jpeg` is a registered zarr extension with implementations in multiple languages, targeting a minimal common API surface.
`imagecodecs_jpeg` supports more features, but is specified only by the details of a single implementation.
