# FlyLab interface specification

Reference: concept.png, generated with the built-in image generation tool. The interface follows the requested left body / right neural activity layout.

## Visual system

Background #101820; surfaces #18232c; borders #2c3842; text #edf2f5; secondary #a0b0bf. Lime #d6ee9b actions, blue #62adff neural signals, orange #ff9a5b selected region and muscles. System sans-serif; 14px controls, 12px captions, 18px panel titles. 8px corner radius; 12px panel gaps. Compact navigation, full-width workbench, 64/36 split, telemetry below.

## Components

Header / experiment toolbar / 3D arena / brain region inspector / stimulation controls / synchronized signal chart / six muscle groups / training panel / model provenance panel. Run, pause, single step, reset, camera views, stimulation and silencing are functional. Training has start, stop, checkpoints and evaluation. Data view presents importer status and model limitations.

## Intentional implementation differences

The fly and neural visualization use interactive geometry, not generated raster art: joints must follow backend state, selected regions must respond to real simulation values. Geometry is schematic, not reconstructed anatomy. Remove invented settings, account and help buttons from the concept. Use six available schematic regions, including descending and motor populations. Training and provenance tabs extend the same component system. Accurate labels take precedence over illustrative values in the concept. Show synthetic reference-model status explicitly. No fabricated connection or training metrics.

## Browser review refinements

Refined the desktop viewport to retain controls and telemetry within a 1536×1024 screen. Point materials use direct signal colors without tone mapping so selection and stimulus highlights remain legible. Added a neutral schematic central brain envelope. All fonts are served locally. Body meshes use attributed research geometry centered and scaled to the schematic rig; this intentionally differs from the photoreal concept. Scale is explicitly uncalibrated.
