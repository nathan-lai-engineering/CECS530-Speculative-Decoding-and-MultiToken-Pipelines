# Adaptive Speculation Depth

## Overview
Scale k speculative depth to the acceptance ratio such that as acceptance ratio increases, speculative depth increases and vice versa. The idea is that we can confidently generate more draft tokens if we expect to accept more of it.

## Acceptance Ratio
Acceptance ratio is the percentage of total draft tokens accepted divided by total draft tokens generated. This helps represent how confident we are in draft tokens being correct.

## Increase Depth
With each speculative batch of k, we increase k by a constant number if acceptance rate is high. We cap the increase to a factor of k so that it does not infinitely grow.

## Decrease Depth
With each speculative batch of k, we divide k by a constant number if acceptance rate is low. We cap the decrease to a minimum of k=2 as 2 is the minimum number depth to be considered as speculation.

## Benefits
- Less verification use on the larger target model
- Overall faster speeds given that acceptance ratio is high enough

## Disadvantages
Low acceptance ratio will trigger the aggressive depth decrease can, in the worst case, reduce k size to 2 which is barely better than the baseline model except with extra overhead.