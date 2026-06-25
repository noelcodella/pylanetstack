# Saturn 16-bit TIFF Example Frames

This folder contains 189 real Saturn capture frames prepared as 16-bit RGB TIFF
crops for quick experiments with PyLanetStack.

The original captures were made with a standard consumer-grade, general-purpose
camera rather than astrophotography-specific hardware such as a dedicated
astronomy camera. Capture setup details and a broader walkthrough are available
in the [Astrophotography Introduction tutorial PDF](https://noelcodella.github.io/publicwebsite/assets/docs/astro-intro.pdf).

Capture parameters from the original ARW files used for this example:

| Camera | Telescope | ISO | Shutter Time(s) |
| --- | --- | --- | --- |
| Sony Alpha 7R IIIA (`ILCE-7RM3A`) | Celestron NexStar 8SE | 100 | `1/10` (189 frames) |

The telescope is a manual optical path, so normal EXIF lens and f-number fields
are not reported by the camera.

Each frame is a 250 x 250 crop centered on Saturn. The crops are stored as
scientific linear 16-bit RGB TIFFs with the original low signal range preserved
for stacking. They may look very dark in generic image previewers; that is
expected. Use the processed stack output or README images for a display-ready
view.

Run the example stack:

```bash
pylanetstack stack examples/data/saturn-01-tiff16 /tmp/pylanetstack-saturn-example/stack.tif \
  --config examples/pylanetstack.planets.yaml
```

The command writes the linear stack, a display preview, a processed TIFF, and
metadata describing which frames were selected. The root README image is based
on a processed stack from this sequence.
