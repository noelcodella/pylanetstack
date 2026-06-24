# Jupiter 16-bit TIFF Example Frames

This folder contains 231 real Jupiter capture frames prepared as 16-bit RGB TIFF
crops for quick experiments with PyLanetStack.

The original captures were made with a standard consumer-grade, general-purpose
camera rather than astrophotography-specific hardware such as a dedicated
astronomy camera. More acquisition background is available in the project
README.

Each frame is a 281 x 281 crop centered on Jupiter. The crops are stored as
scientific linear 16-bit RGB TIFFs with the original low signal range preserved
for stacking. They may look very dark in generic image previewers; that is
expected. Use the processed stack output or README images for a display-ready
view.

Run the example stack:

```bash
pylanetstack stack examples/data/jupiter-2024-11-tiff16 /tmp/pylanetstack-jupiter-example/stack.tif \
  --config examples/pylanetstack.planets.yaml
```

The command writes the linear stack, a display preview, a processed TIFF, and
metadata describing which frames were selected. The root README image is based
on a processed stack from this sequence.
