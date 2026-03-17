# System Model

## Draft Model
A smaller, faster model that predicts k speculative tokens ahead of time.

## Target Model
A larger, more accurate model that verifies the tokens predicted by the draft model.

## Speculation Depth (k)
The number of tokens predicted in advance by the draft model before verification.

## Acceptance Rate
The fraction of speculative tokens that are accepted by the target model during verification.

## Rollback Behavior
Tokens are accepted sequentially until the first mismatch between draft and target outputs.  
All remaining speculative tokens after the mismatch are discarded.
