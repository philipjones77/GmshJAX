# Startup Compile Standard

## Purpose

This standard defines how startup and first-compile behavior should be measured and documented.

## Rules

- measure cold import separately from first runtime construction where possible
- measure first compile or first heavy call separately from steady-state execution
- benchmark outputs should identify whether warmup has already occurred
- startup-sensitive docs should avoid mixing compile cost into ordinary runtime cost without saying so
