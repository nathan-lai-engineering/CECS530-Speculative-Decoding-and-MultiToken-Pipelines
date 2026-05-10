```
INPUT: Prompt p, Draft model draft_model, Target model target_model, Speculation Depth k >= 2, Total Tokens n
OUTPUT: Prompt p with k accepted tokens

WHILE n > 0 DO:
    draft_tokens <- []
    draft_probs <- []
    accepted_tokens <- []
   
    k <- MIN(k, n)
    
    // Generate draft tokens
    FOR i = 1 TO k DO:
        d_p <- draft_model(p + draft_tokens)   
        draft_token <- sample(d_p)
	APPEND(draft_tokens, draft_token)
	APPEND(draft_probs, d_p)                  
    END FOR
        
    target_probs <- target_model(prompt + draft_tokens)
    
    // Perform verification
    FOR i = 0 TO k - 1 DO:
        IF target_probs[i] >= draft_probs[i] THEN:
            APPEND(accepted_tokens, draft_tokens[i])
        ELSE:
	    // Enable rollback on mismatch
            correct_token <- sample(max(0, target_probs[i]- draft_probs[i] // Sample corrected token
            APPEND(accepted_tokens, correct_token)
            CLEAR(draft_tokens) // Discard all remaining draft tokens

            BREAK                              
        END IF
    END FOR
   
    n <- n - LENGTH(accepted_tokens) // Update remaining amount of tokens to be generated

    FOR i = 0 TO LENGTH(accepted_tokens) - 1 DO:
        p <- p + accepted_tokens[i]
    END FOR
     
    CLEAR(accepted_tokens)

END WHILE
RETURN p
```
