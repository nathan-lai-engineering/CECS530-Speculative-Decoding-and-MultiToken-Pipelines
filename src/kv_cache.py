class KVCache:
    """
    Simulates KV-cache behavior for speculative decoding.
    Maintains separate committed and speculative token states.
    """

    def __init__(self):
        self.committed_tokens = []
        self.speculative_tokens = []

    def add_speculative(self, tokens):
        self.speculative_tokens.extend(tokens)

    def verify(self, target_tokens):
        accepted_tokens = []

        for i, (draft_token, target_token) in enumerate(zip(self.speculative_tokens, target_tokens)):
            if draft_token == target_token:
                accepted_tokens.append(draft_token)
            else:
                break
        return accepted_tokens

    def commit(self, accepted_tokens):
        self.committed_tokens.extend(accepted_tokens)

    def rollback(self):
        self.speculative_tokens = []

    def get_state(self):
        return {
            "committed": self.committed_tokens,
            "speculative": self.speculative_tokens
        }
