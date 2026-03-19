# Algorithm Description

## Draft Token Generation
The draft model takes in input from a sentence and generates K tokens for the next K words in the sentence. 

## Parallel Verification
The target model verifies the input sequence and draft tokens by outputting probabilities for each word.

## Acceptance Logic
Both probabilities from the target and draft model are compared. The token is only accepted if
the target model's probability is higher than the draft model's probability.

## Rollback Behavior
If the draft model has a higher probability for a draft token than the target model, the draft token is replaced with a sample from the corrected distribution.
Afterwards, all subsequent tokens are discarded and the new sample is added to the original input sequence and is fed back into the draft model. 
