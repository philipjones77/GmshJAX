# Update Implementation

Common-track updates should be done in this order:

1. update shared specs or standards if the surface is changing
2. update the common module implementation
3. update backend adapters if the common surface moved
4. update reports or status docs that inventory the surface
5. update tests and examples

This keeps the shared layer documented and testable instead of drifting behind the code.
