# Documentation (docs/)

This directory contains the design, algorithm, and architecture specifications for the speculative decoding and multi-token pipeline system.

---

## System Model

### `system_model.md`
Defines the core components of speculative decoding:
- Draft model (fast, predictive)
- Target model (accurate, verification)
- Speculation depth (k)
- Acceptance rate
- Rollback behavior

Tokens are accepted sequentially until a mismatch occurs, after which all remaining tokens are discarded.

---

## Algorithm

### `algorithm_description.md`
Describes the decoding process:
- Draft model generates K tokens
- Target model verifies tokens in parallel
- Tokens are accepted if target probability ≥ draft probability
- On mismatch, token is replaced and remaining tokens are discarded

---

### `pseudocode.md`
Provides a step-by-step version of the algorithm:
- Draft token generation loop
- Parallel verification
- Acceptance check
- Rollback on mismatch

---

## Correctness

### `correctness_proof_sketch.md`
Explains correctness of the method:
- Tokens are only committed after verification
- Mismatches trigger rollback and correction
- Ensures output matches the target model distribution

---

## KV-Cache Strategy

### `kv_cache_strategy.md`
Defines token state management:
- `committed_tokens` → verified tokens
- `speculative_tokens` → temporary tokens

Operations:
- speculative writes
- verification
- commit
- rollback

Ensures only verified tokens persist.

---

## Pipeline Architecture

### `pipeline_analysis.md`
Describes the multi-token pipeline:
- Draft stage (token generation)
- Verification stage (parallel checking)
- Commit / rollback stage

Includes:
- pipeline overlap
- hazards (mismatch, low acceptance)
- buffering and backpressure

---

## Optimizations

### `adaptive_speculative_depth_strategy.md`
Dynamic speculation depth:
- Increase k when acceptance is high
- Decrease k when acceptance is low

---

### `hardware_rollback.md`
Hardware-level rollback concept:
- Speculative buffer
- Commit pointer
- Rollback pointer

Allows fast discard of incorrect tokens without recomputation.

**Note:** This is a conceptual design and is not implemented.

---

## Summary

These documents define:
- The speculative decoding system
- The verification and rollback algorithm
- KV-cache consistency model
- Pipeline execution and hazards
- Performance optimizations
