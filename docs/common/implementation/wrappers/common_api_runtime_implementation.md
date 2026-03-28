# Common API Runtime Implementation

The shared API surface is a wrapper layer over backend providers plus shared helper modules.

Rules:

- common wrappers define stable names and result shapes
- wrappers may delegate immediately to backend implementations
- wrappers should not introduce hidden side effects or broad eager imports

The common layer is meant to reduce discovery cost for users, not to obscure backend ownership.
