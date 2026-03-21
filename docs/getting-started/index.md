# Getting Started

This section is the shortest path from “I just found Truthound” to a real validation run that leaves you with useful artifacts and a clear next step.

## Who This Section Is For

Use Getting Started if you are:

- evaluating Truthound against GX, Pandera, Soda, or a homegrown framework
- onboarding a new teammate
- moving from ad hoc scripts to a repeatable local validation workflow
- trying to understand the 3.0 mental model before reading deeper guides

## Recommended Reading Path

1. [Installation](installation.md)
2. [Quick Start](quickstart.md)
3. [First Validation](first-validation.md)
4. [Zero-Config Context](../concepts/zero-config.md)
5. [Architecture](../concepts/architecture.md)

## Choose Your Workflow

| Workflow | Best first page | When to choose it |
|----------|-----------------|-------------------|
| Python-first | [Quick Start](quickstart.md) | You work in notebooks, scripts, services, or data applications |
| CLI-first | [Quick Start](quickstart.md#cli-workflow) | You want a shell-friendly or CI-friendly path |
| Learn by example | [Tutorials](../tutorials/index.md) | You want end-to-end runnable scenarios instead of reference material |
| Production rollout | [Checkpoints Guide](../guides/checkpoints.md) | You already know the basics and need automation, routing, or notifications |

## What You Will Learn Here

- how to install Truthound and optional extras
- how `th.check()` and `truthound check` behave with zero configuration
- what `.truthound/` contains and why it exists
- when to stay with defaults and when to move into explicit suites, guides, and orchestration

## After Getting Started

- [Tutorials](../tutorials/index.md) for step-by-step learning
- [Guides](../guides/index.md) for task-oriented feature documentation
- [Reference](../reference/index.md) for Python API and CLI lookup
- [Migration to 3.0](../guides/migration-3.0.md) if you are upgrading an existing workflow
