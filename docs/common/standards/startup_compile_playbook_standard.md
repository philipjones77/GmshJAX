# Startup Compile Playbook Standard

## Purpose

This standard defines how startup and compile tuning guidance should be written for the common track.

## Rules

- playbook-style docs should distinguish import cost, runtime construction cost, first compile cost, and steady-state execution
- warmup strategies should say which cache or compile boundary they target
- compile-reduction advice should remain backend-aware
- common wrappers should not claim ownership of backend compile policies they merely expose
