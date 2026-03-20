---
title: Advanced Engines
---

# Advanced Engines

Truthound is the primary engine for the 3.x documentation and validation line. Advanced engines remain supported for teams that need them, but they are a secondary path.

## Supported Advanced-Tier Engines

- Great Expectations
- Pandera
- custom engines registered through the plugin or registry path

## What Changes Compared To Truthound

- documentation depth is lower than the Truthound-first path
- zero-config behavior is centered on Truthound semantics
- compatibility testing focuses first on the first-party Truthound line

## When To Use This Tier

- you already have a mature GE or Pandera investment
- you need a custom engine during migration
- you want to prototype against the shared runtime contracts before standardizing on Truthound
