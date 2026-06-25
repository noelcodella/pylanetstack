# PyLanetStack

PyLanetStack is a Python command line tool for planetary imaging data. It can
start with Sony ARW RAW files, common rendered image files, or common video
captures, find the target in a larger frame, write compact 16-bit TIFF crops,
and stack TIFF/PNG/JPEG/BMP crops with local alignment points plus optional
display processing.

It is designed for planetary, lunar, and solar lucky imaging workflows where
many short exposures are combined to improve signal-to-noise ratio and recover
detail affected by seeing, tracking drift, and chromatic dispersion.

## What It Does

PyLanetStack is meant to make a complete lucky-imaging workflow repeatable
from the command line or a YAML recipe, while still preserving linear
scientific image data when you want to keep processing elsewhere.

1. Runs as a Python implementation on macOS, Windows, and Linux without being tied to one desktop platform.
2. Finds the dominant bright target in every RAW file, image file, or decoded video frame.
3. Saves centered fixed-size or automatically sized square crops as compact 16-bit TIFFs to reduce I/O before stacking.
4. Ranks frames by high-frequency detail, supports best-frame selection, and can weight sharper frames more heavily in the final stack.
5. Applies camera white balance by default so linear RGB crops do not skew green.
6. Aligns each crop with local alignment points for planets, lunar regions, and solar regions.
7. Optionally applies dense nonlinear TV-L1 optical flow for sub-pixel correction of local atmospheric distortion.
8. Corrects atmospheric chromatic aberration/dispersion by aligning RGB color channels, which is especially useful for low-altitude planets.
9. Supports stack-time upsampling for superresolution-style experiments from naturally shifted frames.
10. Saves the best discovered single frame, leveled and processed best-frame comparisons, a linear scientific stack, display previews, processed TIFFs, copied YAML settings, and JSON metadata.

## Example Results

The repository includes compact real-capture example datasets converted into
16-bit TIFF crops so they can be stacked immediately after installation.
The sample captures were acquired with a standard consumer-grade,
general-purpose camera rather than astrophotography-specific hardware such as a
dedicated astronomy camera. That keeps the capture workflow approachable: a
user can collect data without a laptop-tethered planetary camera at the
telescope, and the entry cost can be lower because no specialized imaging
camera is required.
For background on the hardware and software used to acquire the sample data,
see [this astrophotography introduction PDF](https://noelcodella.github.io/publicwebsite/assets/docs/astro-intro.pdf).

| Target | Leveled Best Single Frame | Processed Best Single Frame | Processed Stack |
| --- | --- | --- | --- |
| Mars | ![The best selected Mars frame with levels only](docs/assets/mars-example-single-frame.png) | ![The same best Mars frame with post-processing but no stacking](docs/assets/mars-example-processed-single-frame.png) | ![A processed Mars stack created from the example dataset](docs/assets/mars-example-processed-stack.png) |
| Jupiter | ![The best selected Jupiter frame with levels only](docs/assets/jupiter-example-single-frame.png) | ![The same best Jupiter frame with post-processing but no stacking](docs/assets/jupiter-example-processed-single-frame.png) | ![A processed Jupiter stack created from the example dataset](docs/assets/jupiter-example-processed-stack.png) |
| Saturn | ![The best selected Saturn frame with levels only](docs/assets/saturn-example-single-frame.png) | ![The same best Saturn frame with post-processing but no stacking](docs/assets/saturn-example-processed-single-frame.png) | ![A processed Saturn stack created from the example dataset](docs/assets/saturn-example-processed-stack.png) |
| Venus | ![The best selected Venus frame with levels only](docs/assets/venus-example-single-frame.png) | ![The same best Venus frame with post-processing but no stacking](docs/assets/venus-example-processed-single-frame.png) | ![A processed Venus stack created from the example dataset](docs/assets/venus-example-processed-stack.png) |

The two single-frame columns use the same highest-quality selected frame for
each target. The middle column applies the same post-processing recipe as the
stacked result, but without stacking, so the visual comparison separates
post-processing from the SNR/detail improvement of stacking.

## Table Of Contents

- [Usage Instructions](#usage-instructions)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Input Image Formats](#input-image-formats)
  - [Video Input](#video-input)
  - [YAML Configuration](#yaml-configuration)
  - [Outputs](#outputs)
  - [Recommended Defaults](#recommended-defaults)
- [Advanced Options](#advanced-options)
  - [Cropping And RAW Handling](#cropping-and-raw-handling)
  - [Filtering Frames](#filtering-frames)
  - [Alignment Options](#alignment-options)
  - [Stack-Time Upsampling](#stack-time-upsampling)
  - [Channel Alignment](#channel-alignment)
  - [Post-Processing Options](#post-processing-options)
  - [Preview And Output Format](#preview-and-output-format)
  - [Performance And Threads](#performance-and-threads)
  - [Common Workflows](#common-workflows)
  - [Development](#development)

## Usage Instructions

### Installation

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`rawpy` is required for Sony ARW input. `imageio-ffmpeg` is installed for common
video decoding. The stacker itself operates on cropped image frames, so
`pylanetstack stack` can process existing TIFF, PNG, JPEG, or BMP crops from any
camera when RAW support or video decoding is not needed.

### Quick Start

Run the whole crop + stack pipeline:

```bash
pylanetstack run /path/to/input-folder /path/to/output-folder --overwrite
```

The older `planetstack` command is still installed as a compatibility alias.

That creates:

```text
/path/to/output-folder/crops/*.tif
/path/to/output-folder/crops/crops.jsonl
/path/to/output-folder/crops/crop_errors.jsonl  # only if RAW files are skipped
/path/to/output-folder/stack.tif
/path/to/output-folder/stack_preview.tif
/path/to/output-folder/stack_processed.tif
/path/to/output-folder/stack_best_frame.tif
/path/to/output-folder/stack_best_frame_leveled.tif
/path/to/output-folder/stack.json
```

Crop only:

```bash
pylanetstack crop /path/to/input-folder /path/to/crops --crop-size 250
```

Restack existing crops without re-reading source files:

```bash
pylanetstack stack /path/to/output-folder/crops /path/to/output-folder/stack.tif
```

For a faster first trial on either still images or a video:

```bash
pylanetstack run /path/to/input-folder /path/to/output-folder --max-frames 20
```

Omit `--max-frames 20` to process the full folder.

Run an included example dataset:

```bash
pylanetstack stack examples/data/mars-01-tiff16 /tmp/pylanetstack-mars-example/stack.tif \
  --config examples/pylanetstack.planets.yaml
```

Additional bundled planetary examples:

```bash
pylanetstack stack examples/data/jupiter-2024-11-tiff16 /tmp/pylanetstack-jupiter-example/stack.tif \
  --config examples/pylanetstack.planets.yaml

pylanetstack stack examples/data/saturn-01-tiff16 /tmp/pylanetstack-saturn-example/stack.tif \
  --config examples/pylanetstack.planets.yaml

pylanetstack stack examples/data/venus-01-tiff16 /tmp/pylanetstack-venus-example/stack.tif \
  --config examples/pylanetstack.planets.yaml
```

These use the shared planets recipe with automatic window levels. The bundled
TIFF crops preserve low-range linear data for stacking, so they may look dark in
generic previewers. The example images above correspond to processed stacks;
`stack_preview.tif` is an automatic percentile stretch of the linear stack and
may look brighter.

### Input Image Formats

`pylanetstack crop` and `pylanetstack run` can ingest Sony ARW RAW files,
already rendered full-frame image files, or common video files. They write
16-bit TIFF crops.

`pylanetstack stack` reads already-cropped image frames. Supported input
extensions are `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, and `.bmp`. TIFF and
16-bit PNG are preferred for serious stacking. JPEG and 8-bit PNG/BMP can be
useful for tests or third-party exports, but they have already lost bit depth
and, for JPEG, may contain compression artifacts.

When 8-bit images are cropped or stacked, PyLanetStack expands their values
into the 16-bit numeric range on read so the default preview and post-processing
levels behave like the TIFF workflow. This preserves relative values, but it
cannot recover detail that was not present in the source file.

### Video Input

Video input is handled during `pylanetstack crop` and `pylanetstack run`.
Supported filename extensions are `.mov`, `.mp4`, `.m4v`, `.avi`, `.mkv`, and
`.webm`, subject to FFmpeg codec support. Each decoded frame becomes a normal
TIFF crop named with its source frame index, such as
`capture_frame000120_crop.tif`.

Use `--video-frame-step` to sample long captures:

```bash
pylanetstack run capture.mp4 /path/to/output-folder --video-frame-step 5 --overwrite
```

That keeps frames 0, 5, 10, and so on. `--max-frames` limits the total number
of crops written, so it also works as a quick trial limit for video captures.

`pylanetstack stack` does not read videos directly. Use `run` for the full
workflow or `crop` first, then restack the generated TIFF crop folder.

### YAML Configuration

PyLanetStack can read grouped YAML settings so repeatable recipes do not need
long command lines. The precedence order is:

1. Hard-coded defaults.
2. A default YAML file in the current directory.
3. A YAML file passed with `--config`.
4. Command-line flags.

Command-line flags always win over YAML.

When a YAML file is loaded, PyLanetStack copies it into the output folder as
`pylanetstack_used_config.yaml`. This preserves the recipe that started the run;
any command-line overrides are still recorded separately in `stack.json`.

Default config file names are checked in this order:

```text
pylanetstack.yaml
pylanetstack.yml
.pylanetstack.yaml
.pylanetstack.yml
planetstack.yaml
planetstack.yml
.planetstack.yaml
.planetstack.yml
```

Start from the example:

```bash
cp pylanetstack.example.yaml pylanetstack.yaml
```

Or start from a workflow recipe:

```bash
pylanetstack run INPUT OUTPUT --config examples/pylanetstack.planets.yaml --overwrite
pylanetstack run INPUT OUTPUT --config examples/pylanetstack.moon.yaml --overwrite
```

Then run normally:

```bash
pylanetstack run INPUT OUTPUT --overwrite
```

Or pass a config explicitly:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif --config planet.yaml
```

Disable automatic config loading:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif --no-config
```

Example YAML:

```yaml
performance:
  workers: 8

crop:
  crop_size: auto
  auto_crop_margin: 1.35
  auto_crop_min_size: 250
  auto_crop_sample_frames: 12
  video_frame_step: 1

filtering:
  keep_best_percent: 10
  quality_weight_scale: 1.5

alignment:
  method: ap
  registration_max_shift: 50
  ap:
    patch_size: 12
    spacing: 6
    blend_sigma: 18

channel_alignment:
  enabled: true
  max_shift: 3

output:
  stack_upsample: 1

postprocessing:
  level_black: auto
  level_white: auto
  sharpening: 500
  wb_strength: 1
  saturation: 3
  vibrance: 1
  upsample: 4
  upsample_smoothing: 1.0
```

### Outputs

`stack.tif` is the linear scientific stack. It defaults to 16-bit integer output
so ordinary image viewers do not treat it as white float data. Use
`--output-dtype float32` to preserve sub-integer averaged values. Use
`--stack-upsample` to accumulate the scientific stack on a larger grid.

`stack_preview.tif` is a simple display-stretched TIFF for quick inspection.

`stack_processed.tif` is the display/export version. It applies an adjustable
recipe: levels, exposure, highlight compression, texture, dehaze, sharpening,
processed white balance, saturation/vibrance, background feathering, and
optional display upsampling.

`stack_best_frame.tif` is the sharpest single crop that survived frame
filtering and was actually used in the stack.

`stack_best_frame_leveled.tif` is that same best frame with only the resolved
levels/gamma applied. It is meant as a fair, visible single-frame comparison
without texture, dehaze, sharpening, color boost, or background masking.

`stack_best_frame_processed.tif` applies the same full postprocessing recipe to
that one frame, which makes it easy to compare the best single-frame result
with the final processed stack.

`stack.json` records the stack settings, selected and accepted frames, rejected
registration outliers, weights, shifts, alignment-point counts, channel shifts,
best-frame paths, and output statistics.

`pylanetstack_used_config.yaml` is a copy of the YAML file that launched the
command, when one was loaded.

### Recommended Defaults

The default settings are a practical starting point for small planetary targets:

```bash
pylanetstack run INPUT OUTPUT --overwrite
```

These defaults are also shown in `pylanetstack.example.yaml` and match
`examples/pylanetstack.planets.yaml`, so the planets YAML is optional unless you
want an explicit, portable recipe file. `examples/pylanetstack.moon.yaml` remains
a specialized starting point for larger lunar-style frames. The default planet
recipe uses automatic display levels with extra highlight headroom for compact
bright disks, plus automatic post-process white balance to keep bright details
neutral while preserving planetary color. The command above is equivalent to
the important parts of:

```bash
pylanetstack run INPUT OUTPUT \
  --overwrite \
  --workers 8 \
  --video-frame-step 1 \
  --method ap \
  --keep-best-percent 25 \
  --quality-weight-scale 1.5 \
  --channel-align \
  --ap-patch-size 12 \
  --ap-spacing 6 \
  --ap-blend-sigma 18 \
  --ap-max-points 2500 \
  --flow-max-pixels 2 \
  --flow-smoothing-sigma 0.1 \
  --stack-upsample 1 \
  --postprocess \
  --post-level-black auto \
  --post-level-white auto \
  --post-exposure 0.5 \
  --post-highlight-compression 0.7 \
  --post-texture 100 \
  --post-dehaze 100 \
  --post-sharpening 500 \
  --post-wb-mode auto \
  --post-wb-strength 1 \
  --post-saturation 3 \
  --post-vibrance 1 \
  --post-background-black \
  --post-upsample 4 \
  --post-upsample-smoothing 1.0
```

`--overwrite` remains explicit so an existing crop folder is not replaced by
accident.

## Advanced Options

### Cropping And RAW Handling

For RAW inputs, the crop step defaults to `linear-rgb`. It reads the RAW Bayer
mosaic, detects the planet without demosaicing the full sensor, crops the local
raw sensor region, and bilinear-demosaics only that small crop. Output TIFFs are
16-bit and are not gamma encoded, auto-brightened, or JPEG compressed.

For TIFF, PNG, JPEG, and BMP inputs, the crop step uses image luminance to find
the planet and writes a centered TIFF crop from the rendered pixels. Existing
16-bit pixels are preserved; 8-bit pixels are expanded into the 16-bit numeric
range.

For video inputs, frames are decoded one at a time, cropped, and immediately
written as TIFFs. This keeps memory use steady even for long high-frame-rate
captures.

Cropping options:

- `--crop-size`: fixed square crop size in pixels, or `auto`. Default `250`.
- `--auto-crop-margin`: multiplier around the detected target bounding box when `--crop-size auto` is used. Default `1.35`.
- `--auto-crop-min-size`: minimum automatic square crop size. Default `250`, so small planets keep a compact crop size.
- `--auto-crop-max-size`: optional maximum automatic square crop size. By default, auto crops are capped at the larger source-frame dimension.
- `--auto-crop-sample-frames`: number of frames to inspect when choosing one automatic crop size for the whole dataset. Default `12`.
- `--video-frame-step`: for video inputs, keep every Nth decoded frame. Default `1`.
- `--recursive`: search nested folders for RAW/image/video files.
- `--max-frames`: process only the first N crop frames for a quick trial. For videos, this limits decoded frames that are actually written after `--video-frame-step`.
- `--overwrite`: replace existing crop/output files.
- `--save-raw-crop`: also save the raw Bayer mosaic crop.
- `--output-mode linear-rgb`: write locally demosaiced RGB crops. This is the default.
- `--output-mode raw-mosaic`: write literal Bayer mosaic crops. The stacker expects RGB or grayscale image crops, so this is mainly for preservation/debugging.
- `--white-balance camera`: apply camera white-balance metadata. This is the default.
- `--white-balance auto`: estimate crop white balance from the planet signal itself.
- `--white-balance none`: keep unbalanced sensor RGB.
- `--skip-bad-raw` / `--no-skip-bad-raw`: skip unreadable RAW files by default, or fail fast in strict mode.

Auto crop mode samples frames across the dataset, estimates the crop size each
sample needs, then uses the largest sampled size as one universal square crop
for every output frame. Per-frame target centers can still move, but the crop
dimensions stay fixed for stable alignment and stacking. The estimator uses an
Otsu-style foreground threshold on a downsampled luminance/raw signal, finds the
dominant contiguous target region, projects that region along X and Y, and adds
the requested margin. This is intended for larger solar/lunar targets and other
cameras where a fixed 250px crop is too small.

Example:

```bash
pylanetstack run INPUT OUTPUT \
  --crop-size auto \
  --auto-crop-margin 1.35 \
  --auto-crop-sample-frames 12 \
  --overwrite
```

`--output-mode raw-mosaic`, `--white-balance`, and `--skip-bad-raw` are
RAW-specific. Rendered image inputs and decoded video frames are cropped as-is
after any 8-bit to 16-bit numeric scaling.

If LibRaw/rawpy cannot read one ARW file, the crop step prints a `[crop:skip]`
warning, continues the batch, and records the failure in `crop_errors.jsonl`.

Exact sub-pixel centering requires interpolation, so the crop is centered to the
nearest source pixel and records the measured sub-pixel planet center in
`crops.jsonl`. The stack stage performs sub-pixel and nonlinear registration.

### Filtering Frames

Planetary stacks often improve when soft frames are removed instead of merely
downweighted.

Frame-selection options:

- `--keep-best-percent`: stack only the sharpest percentage of frames. Default `25`.
- `--keep-best-frames`: stack only the sharpest N frames.
- `--quality-weighting` / `--no-quality-weighting`: weight frames by high-frequency detail, or give each crop equal weight.
- `--quality-weight-scale`: strength of sharp-frame weighting. Default `1.5`; `0` makes all weights equal.
- `--quality-weight-min` and `--quality-weight-max`: clamp frame weights so one very sharp or noisy frame cannot dominate.
- `--reference best`: align to the sharpest selected crop. This is the default.
- `--reference first`: align to the first selected crop.

Example:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --keep-best-percent 10 \
  --quality-weight-scale 1.5
```

### Alignment Options

The default alignment method is `ap`: estimate a global sub-pixel shift, lay
overlapping alignment points over the planet, measure local patch shifts, and
blend those into one smooth warp.

Alignment methods:

- `--method ap`: local alignment-point registration. This is the default.
- `--method rigid`: global sub-pixel translation only.
- `--method flow`: dense TV-L1 optical flow after global translation.
- `--method none`: no frame-to-frame registration.

Alignment-point options:

- `--ap-patch-size`: alignment-point patch size in pixels. Default `12`.
- `--ap-spacing`: distance between AP centers. Default `6`.
- `--ap-blend-sigma`: blend radius used to turn AP shifts into a smooth warp. Default `18`.
- `--ap-max-points`: maximum AP points per frame. Default `2500`; use `0` to disable the cap.
- `--ap-min-mask-fraction`: reject AP patches that contain too little planet signal.
- `--ap-min-contrast`: reject AP patches with too little local contrast.
- `--registration-max-shift`: reject selected frames whose global registration shift exceeds this many pixels. Default `50`; use `0` to disable. This catches sharp frames from a different framing group before they can create a ghost object in the stack. For color lunar/planetary data, leave channel alignment enabled so the shift estimate is based on the cleaned-up channels.

For large lunar or solar crops, the small-planet AP patch and spacing defaults
would otherwise create an enormous candidate grid. PyLanetStack automatically
widens AP spacing when the candidate grid exceeds `--ap-max-points`, then
prints the effective spacing, accepted AP shift count, per-frame elapsed time,
and ETA. For finer local lunar alignment, raise `--ap-max-points` and use
larger AP patches/spacings, for example:

```bash
--ap-patch-size 96 --ap-spacing 48 --ap-blend-sigma 96 --registration-max-shift 50
```

Flow regularization options:

- `--flow-max-pixels`: maximum local nonlinear displacement. Default `2`; use `0` for no cap.
- `--flow-smoothing-sigma`: Gaussian smoothing applied to the flow field before warping. Default `0.1`.
- `--flow-warps`, `--flow-iterations`, `--flow-attachment`, `--flow-tightness`: lower-level TV-L1 optical-flow controls.

Reference update option:

- `--update-reference-every`: refresh the alignment reference from the running stack every N frames.

Example AP stack:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --method ap \
  --ap-patch-size 12 \
  --ap-spacing 6 \
  --ap-blend-sigma 18 \
  --ap-max-points 2500
```

Example constrained optical-flow experiment:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --method flow \
  --flow-max-pixels 2 \
  --flow-smoothing-sigma 0.1
```

### Stack-Time Upsampling

`--stack-upsample` enlarges the actual scientific stack during accumulation.
This is different from `--post-upsample`, which only enlarges the processed
display image after stacking.

When frames contain real sub-pixel shifts, stack-time upsampling can behave like
a simple drizzle/superresolution stack: each registered frame is sampled onto a
larger output grid using its measured global shift and, for `ap`/`flow`, its
local warp field. This can make fine detail look less blocky than display-only
resizing. It cannot recover detail blurred away by seeing, focus, compression,
or heavy preprocessing, but it is the better experiment when post-processed
upsampling looks jagged.

Options:

- `--stack-upsample`: scientific stack scale. Default `1`; try `1.5` or `2`.
- `--post-upsample`: display-only processed-output scale. Keep this at `1` when using `--stack-upsample` unless additional display enlargement is desired.

Example 2x stack-time upsampling:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --stack-upsample 2 \
  --post-upsample 1
```

### Channel Alignment

Atmospheric dispersion can shift red and blue relative to green. Channel
alignment happens before frame-to-frame registration, and a second cleanup pass
is applied to the final stack because the higher-SNR stack can reveal small
residual red/blue offsets that were hard to estimate in individual noisy frames.

Options:

- `--channel-align` / `--no-channel-align`: align RGB channels to a reference channel. Enabled by default.
- `--channel-reference`: reference color channel. Default `green`.
- `--channel-align-max-shift`: cap chromatic channel shifts in pixels. Default `3`.
- `--channel-align-upsample-factor`: sub-pixel precision factor for channel shifts. Default `20`.

Example:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --channel-align \
  --channel-align-max-shift 3
```

### Post-Processing Options

Post-processing writes `stack_processed.tif`. It does not modify the scientific
`stack.tif`.

Enable or disable:

- `--postprocess` / `--no-postprocess`: write or skip `*_processed.tif`.

Tone controls:

- `--post-input-max`: input value treated as white before the Photoshop-style levels step. Default `65535`.
- `--post-level-black`: input black level on a 0-255 scale, or `auto`. Default `auto`.
- `--post-level-white`: input white level on a 0-255 scale, or `auto`. Default `auto`.
- `--post-level-gamma`: levels gamma. Default `1`.
- `--post-exposure`: exposure in stops. Default `0.5`.
- `--post-highlight-compression`: soft highlight rolloff after exposure. Default `0.7`; use `0` to disable.

`auto` levels use robust luminance percentiles from the stack, measured around
the bright target when possible, then add adaptive highlight headroom. Small,
dim planets get more headroom to avoid clipped color channels; larger bright
surfaces get a tighter but still conservative white point. The requested values
and the resolved numeric levels are both written to `stack.json`, so a run
remains reproducible after tuning.

Detail controls:

- `--post-texture`: mid-scale luminance contrast. Default `100`.
- `--post-texture-radius`: blur radius used for texture detail. Default `2`.
- `--post-dehaze`: broad local contrast and black recovery. Default `100`.
- `--post-dehaze-radius`: blur radius used for dehaze detail. Default `12`.
- `--post-sharpening`: luminance unsharp-mask strength. Default `500`; higher values are allowed for tuning.
- `--post-sharpening-radius`: unsharp-mask radius. Default `0.8`.

Color controls:

- `--post-wb-mode auto`: measure planet disk midtones and partially neutralize broad color casts. This is the default.
- `--post-wb-mode none`: skip automatic processed-output white balance.
- `--post-wb-strength`: auto white-balance strength from `0` to `1`. Default `1`.
- `--post-red-balance`, `--post-green-balance`, `--post-blue-balance`: manual channel multipliers.
- `--post-saturation`: processed-output color multiplier. Default `3`.
- `--post-vibrance`: extra lift for weakly saturated pixels. Default `1`.

Background controls:

- `--post-background-black` / `--no-post-background-black`: feather the processed output to black outside the detected planet. Enabled by default.
- `--post-background-dilate`: expand the detected planet mask before feathering. Default `3`.
- `--post-background-feather`: soften the background mask edge. Default `4`.

Display size control:

- `--post-upsample`: enlarge `stack_processed.tif` by this scale factor. Default `4`; use `1` to keep the processed output at native stack size. This does not change `stack.tif` or `stack_preview.tif`.
- `--post-upsample-smoothing`: smoothing sigma used during processed-output upsampling. Default `1.0`; use `0` for the crispest result or `0.5` to `1.0` for a softer, more natural display image.

When upsampling is enabled, PyLanetStack enlarges the tone-mapped image before
texture, dehaze, sharpening, and background feathering. That generally looks
smoother than enlarging after sharpening because the detail filters operate at
the final display size.

Example: reduce a green/yellow cast:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --post-wb-strength 1.0
```

Example: manual color tweak:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --post-green-balance 0.95 \
  --post-blue-balance 1.08
```

Example: test stronger sharpening:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --post-sharpening 700
```

Example: write a 2x larger processed display image:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --post-upsample 2
```

Example: write a softer 2x processed display image:

```bash
pylanetstack stack INPUT_CROPS OUTPUT/stack.tif \
  --post-upsample 2 \
  --post-upsample-smoothing 0.6
```

### Preview And Output Format

- `--stack-upsample`: write `stack.tif`, `stack_preview.tif`, and `stack_processed.tif` on an enlarged scientific stack grid. Default `1`.
- `--output-dtype uint16`: write the linear stack as 16-bit integers. This is the default.
- `--output-dtype float32`: preserve sub-integer averaged stack values.
- `--preview` / `--no-preview`: write or skip `*_preview.tif`.
- `--preview-low-percentile`: low percentile for preview stretch. Default `0.5`.
- `--preview-high-percentile`: high percentile for preview stretch. Default `99.95`.
- `--compression`: TIFF compression for `stack`.
- `--crop-compression` and `--stack-compression`: separate TIFF compression controls for `run`.

### Performance And Threads

Use `--workers N` to run crop processing, quality scoring, and fixed-reference
stack registration on multiple worker threads. The default is `8`, which is a
practical starting point for modern laptops. Use `--workers 1` for conservative
serial execution, or `--workers 0` to request all CPU cores.

Threads share the reference image, so this is much more memory-friendly than
process-based parallelism. Still, large lunar or solar crops can be several
thousand pixels across, and each stack worker may hold multiple large float
arrays while it registers a frame. For large datasets, try `--workers 2` or
`--workers 4` first before using all cores.

If `--update-reference-every` is enabled, stack registration automatically falls
back to one worker because the reference changes during accumulation.

YAML:

```yaml
performance:
  workers: 4
```

### Common Workflows

Rerun only stacking/post-processing after crops already exist:

```bash
pylanetstack stack /path/to/output-folder/crops /path/to/output-folder/stack.tif
```

Re-crop and restack a source folder:

```bash
pylanetstack run /path/to/input-folder /path/to/output-folder --overwrite
```

Strict run that aborts on the first unreadable RAW:

```bash
pylanetstack run /path/to/raw-folder /path/to/output-folder --overwrite --no-skip-bad-raw
```

### Development

```bash
python -m pytest
```

The tests use synthetic planetary frames so they do not require camera files.

## Development Note

This repository was developed with assistance from OpenAI Codex.
