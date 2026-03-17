# Adaptive Speculation Depth

## Overview
Scale k speculative depth to the acceptance ratio such that as acceptance ratio increases, speculative depth increases and vice versa. The idea is that we can confidently generate more draft tokens if we expect to accept more of it.

## Acceptance Ratio
Acceptance ratio is the percentage of total draft tokens accepted divided by total draft tokens generated. This helps represent how confident we are in draft tokens being correct.

## Increase Depth
With each speculative batch of k, we increase k by a constant number if acceptance rate is high

## Decrease Depth
With each speculative batch of k, we divide k by a constant number if acceptance rate is low

## Benefits
- Less verification use on the larger target model
- Overall faster speeds given that acceptance ratio is high enough

