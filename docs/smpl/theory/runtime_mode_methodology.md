# SMPL Runtime Mode Methodology

SMPL exposes both direct runtime evaluation and mode-oriented optimization surfaces because repeated body-model use has two practical needs:

- fast direct forward evaluation
- structured optimization workflows with diagnostics, artifacts, and bridge/export surfaces

The optimized runtime therefore makes cache, compile, and padding policy explicit instead of burying those choices behind an opaque forward call.
