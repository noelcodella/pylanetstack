# Algorithm Notes

## RAW crop detection

PyLanetStack locates the planet on the RAW Bayer mosaic, not on a fully
demosaiced full-resolution image. It downsamples the mosaic with block means,
smooths the result, estimates the background with robust statistics, and picks
the connected bright component with the strongest total signal. The centroid is
then refined on the original-resolution mosaic inside a local window.

This keeps the full-frame operation inexpensive and avoids producing a large
temporary RGB image for every ARW file.

## Crop output

The default `linear-rgb` crop writes a 16-bit TIFF after local bilinear
demosaicing. It applies camera white balance by default by scaling signal above
the camera black level; this corrects the green-heavy raw sensor response while
preserving a linear intensity scale. It does not apply gamma, auto brightness,
tone curves, or lossy compression. The measured source centroid, integer crop
origin, crop size, white-balance mode, camera black/white levels, and output path
are written to a JSON Lines manifest.

The `raw-mosaic` crop mode writes the cropped Bayer samples directly.

## Registration

Stacking starts with a reference crop, usually the crop with the highest
high-frequency detail score. Each incoming crop is normalized only for
registration, then aligned with phase correlation by default.

The optional `ap` method uses local alignment-point registration: after the
global sub-pixel shift, PyLanetStack lays overlapping alignment points over the
planet, measures a residual sub-pixel shift for each patch, caps those local
shifts, and blends them into one smooth displacement field. This allows belts,
limbs, and storms to move slightly differently without giving every pixel the
freedom of dense optical flow.

The optional `flow` method uses two stages:

1. Phase correlation estimates a global sub-pixel translation.
2. TV-L1 optical flow estimates a small local warp field.

The optical-flow field is regularized before warping. `--flow-smoothing-sigma`
applies Gaussian smoothing to encourage a smooth displacement field, and
`--flow-max-pixels` clips each local displacement vector to a maximum magnitude
so the warp cannot move surface details unrealistically far.

The original linear pixel values are warped and accumulated; the normalized
registration images are not used as stack values. The default `stack.tif` is
written as a 16-bit linear TIFF so ordinary image viewers do not clip float data
to white. Use `--output-dtype float32` when sub-integer averaged values are more
important than viewer compatibility.

## Post-processing

PyLanetStack writes an optional `stack_processed.tif` by default. This is not a
replacement for the linear stack; it is a display/export product that approximates
a useful Photoshop/Camera Raw workflow. The steps are Photoshop-style input
levels on a 0-255 scale, exposure in stops, a soft highlight shoulder, luminance
texture enhancement, dehaze-like broad local contrast, luminance unsharp masking,
processed-output white balance, explicit saturation/vibrance recovery, and
optional feathering to black outside the detected planet. The automatic white
balance measures planet disk midtones and partially neutralizes broad color casts
before color is boosted, while manual RGB balance multipliers allow by-eye
matching to another renderer such as Photoshop. The color recovery exists
because the linear stack can have very weak chroma even when Photoshop Camera Raw
would render more color after dehaze and profile adjustments. Sharpening
intentionally accepts values above 100 so a user can oversharpen while searching
for the best level. Each stage can be adjusted, and the whole processed output
can be disabled with `--no-postprocess`.

## Channel alignment

Atmospheric dispersion can shift blue and red light relative to green. When
`--channel-align` is enabled, each crop's RGB channels are aligned to a reference
channel, green by default, before frame-to-frame registration. The channel shifts
are sub-pixel phase-correlation shifts and are capped by
`--channel-align-max-shift` to avoid chasing noise or color artifacts.

## Quality weighting

The quality score is the RMS Laplacian energy inside a robust planet mask. Scores
are divided by the dataset mean, raised to `--quality-weight-scale`, normalized
back near mean weight 1, and clipped with `--quality-weight-min` and
`--quality-weight-max` to avoid letting a single frame dominate. This favors
frames with stronger fine detail while keeping the stack stable.

Weighting does not remove soft frames; it only reduces their influence. For
planetary lucky imaging, `--keep-best-percent` or `--keep-best-frames` can reject
low-quality frames completely before alignment and accumulation. The selected
frames keep their original chronological order after quality culling.

## Memory behavior

The crop stage processes one ARW at a time and only copies the local crop. The
stack stage reads one cropped TIFF at a time and maintains the reference, the
current frame, and one float64 accumulator. For 250 x 250 RGB crops this is tiny
relative to a 64 GB MacBook Pro.
