# Common Mesh Movement Modes

The common layer uses one shared vocabulary for mesh-movement modes so backend capabilities can be compared and dispatched consistently.

The mode system matters in the common track because:

- backend reports need a shared language
- bridge payloads need stable mode identity
- NumPy runtimes need to mirror the same mode categories even without AD

The common layer does not redefine backend algorithms. It standardizes how those algorithms are named and surfaced.
