# Topo Runtime Surface Methodology

Topo uses a workflow-oriented surface because mesh optimization is more than a single pure function call.

The runtime surface is designed to preserve:

- domain initialization context
- mode identity
- optimization history
- exportable artifacts
- bridge payload generation

That is why the practical surface favors structured workflow runs and artifact bundles rather than only exposing raw optimized coordinates.
