# Multi-Token Pipeline Architecture

## Pipeline Overview
The system consists of two main stages: draft generation and verification.

## Draft Stage
The draft model generates k speculative tokens and stores them in a temporary buffer.

## Verification Stage
The target model verifies the speculative tokens and determines the accepted prefix.

## Overlap of Execution
While the target model verifies the current batch of tokens, the draft model can continue generating future tokens.

## Pipeline Stages
1. Draft token generation  
2. Speculative storage  
3. Verification  
4. Commit or rollback  

## Pipeline Hazards
- Mismatch between draft and target tokens causes rollback  
- Low acceptance rate reduces efficiency  

## Buffering and Backpressure
If the verification stage is slower than the draft stage, speculative tokens accumulate and must be controlled.

## Resolution
Rollback ensures correctness by discarding incorrect speculative tokens and committing only verified tokens.
