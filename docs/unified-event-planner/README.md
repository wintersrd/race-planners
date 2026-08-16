# Unified Event Planner

Date: 2026-07-02
Status: Active handoff documentation for the unified planner build

## Purpose

This folder captures the current product direction for merging the legacy
semi-marathon planner and the newer general planner into a single event-first
planning experience.

These notes are intended to let a later implementation agent resume work
without having to repeat the discovery session, reverse-engineer the existing
code split, or infer product decisions from scattered conversation history.

## Documents

- `research-notes.md`
  - repository and codebase discovery
  - strengths and weaknesses of both existing planners
  - GPX and course-library findings
  - explicit user requirements captured during the interview
- `product-and-architecture.md`
  - merged product definition
  - unified user flow
  - event catalog and template model
  - target domain model and architectural direction
- `implementation-plan.md`
  - phased build plan
  - sequencing and risk reduction
  - testing and validation requirements

## Scope Summary

The target is not "make the beta planner a little nicer." The target is a
single planning tool where the user picks a curated event and gets:

- the correct pacing model automatically
- the correct GPX/course automatically
- either finish-time or effort-anchor input
- the stronger legacy-style visual experience
- trail and ultra-specific controls only when relevant
- dynamic section guidance derived from aid-station boundaries and elevation

## Historical Context

Older notes under `docs/ultra-planner/` describe the earlier general-planner
direction. They remain useful as background, but they are no longer the
authoritative direction for the next build stage because the product scope has
changed in several material ways:

- the planners should be merged into one tool
- the experience should be event-first, not planner-mode-first
- custom GPX upload is out of scope for the merged curated flow
- GRF events are now in scope and should map to trail-ultra behavior
- legacy UX strengths should be preserved in the merged experience

## Immediate Next Build Goal

Use these docs to implement the first unified vertical slice:

- event catalog abstraction
- single event selector UI
- one canonical planner flow
- support for Finistere half marathon and GRF events in the same app shell

Do not start by adding more planner toggles. There are already enough ways to
hide the architecture problem behind UI switches.
