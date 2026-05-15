# ADR-001: Platform Tagging Strategy

**Status:** Accepted  
**Date:** 2026-05

## Context

The torch-spyre repository needed a systematic way to distinguish
issues and PRs targeting different hardware layers:

1. IBM POWER/Z host CPU work
2. Spyre AIU card work (firmware, compiler, driver)
3. x86 host work

## Decision

Introduce three platform labels:

- `platform: power-z` — POWER/Z host work
- `platform: spyre-aiu` — AIU card work
- `platform: x86` — x86-specific work (applied sparingly)

Plus a binary migration tag: `is_power_z_dev`

## Consequences

- Enables quantitative tracking of POWER/Z platform maturity
- Supports research-grade signal collection for team maturity analysis
- Forward-only: no retrospective tagging required
- Analytical horizon starts from label adoption date
