# Correctness Proof Sketch

## Draft Model
The draft model takes in input from a sentence and generates K tokens of what it thinks the next K words should be in the sentence. It also outputs
a probability for each word on how confident that it is the next word.

## Target Model
The target model verifies the input sequence and draft tokens and outputs probability distributions 
for each draft token that ranges over every possible words in its vocabulary. The highest probability out of that distribution is used because it is the word
that the target model is most confident on.

## Acceptance Logic
Each probability from the draft model is compared with the probability from the target model over which draft tokens to accept or reject. The token is only accepted if
the target model's probability is higher than the draft model's probability because the probability has to match the target model's probability distribution. This means that
it has to be a word that the target model would most likely generate.

## Rollback Behavior
If the draft model has a higher probability for a draft token than the target model, the draft token is replaced with a sample from the corrected distribution. This helps to correct both models so that 
it has a higher chance of knowing what draft token to generate after that corrected token. Afterwards, all subsequent tokens are discarded because it would result in a different sequence than what the target model outputs. 
This new sample is added to the original input sequence and is fed back into the draft model. 