INPUT: prompt, draft_model, target_model, K (number of draft tokens), current_n

REPEAT until current_n EQUALS 0:
    draft_tokens = []
    draft_probs = []
    accepted_tokens = []
    
    # Generate K draft tokens
    FOR i = 1 to K:
        p_draft = draft_model(prompt + draft_tokens)   
        token = sample(p_draft)                  
        draft_tokens.append(token)
        draft_probs.append(p_draft)

    # Generate probabilities for draft tokens from target model
    target_probs = target_model(prompt + draft_tokens)
    
    FOR i = 0 to K - 1:
        # Verify each draft token
        p_target = target_probs[i]
        p_draft  = draft_probs[i]

        IF p_target >= p_draft:
            accepted_tokens.append(draft_tokens[i])
            current_n -= 1

        ELSE:
            new_token = sample(max(0, p_target - p_draft))
            accepted_tokens.append(new_token)
            BREAK                              

    FOR i = 0 to LENGTH(accepted_tokens) - 1:
        prompt = prompt + accepted_tokens[i]