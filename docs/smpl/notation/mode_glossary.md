# SMPL Mode Glossary

Key SMPL runtime terms:

- baseline runtime: the direct `SMPLJAXModel` surface
- optimized runtime: `OptimizedSMPLJAX` with explicit cache and compile policy
- runtime mode: the creation-time selection of baseline or optimized behavior
- batch bucket: a padded batch-size target used to reduce recompiles
- fixed padded batch size: a single compile shape reused by smaller requests
- phase summary or group summary: structured metadata returned by higher modes
