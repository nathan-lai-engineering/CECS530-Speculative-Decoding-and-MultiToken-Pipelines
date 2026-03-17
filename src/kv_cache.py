class KVCache:
    """
    Simulates KV-cache behavior for speculative decoding.
    Maintains separate committed and speculative token states.
    """

    def __init__(self):
        self.committed_tokens = []
        self.speculative_tokens = []

    def add_speculative(self, tokens):
        print(f"[Draft] Adding speculative tokens: {tokens}")
        self.speculative_tokens.extend(tokens)
        print(f"[State] Speculative = {self.speculative_tokens}\n")

    def verify(self, target_tokens):
        print(f"[Verify] Target tokens: {target_tokens}")
        accepted_tokens = []

        for i, (draft_token, target_token) in enumerate(zip(self.speculative_tokens, target_tokens)):
            print(f"  Compare position {i}: draft={draft_token}, target={target_token}")
            if draft_token == target_token:
                accepted_tokens.append(draft_token)
            else:
                print(f"  MISMATCH at position {i} -> stopping verification")
                break

        print(f"[Verify] Accepted prefix: {accepted_tokens}\n")
        return accepted_tokens

    def commit(self, accepted_tokens):
        print(f"[Commit] Committing tokens: {accepted_tokens}")
        self.committed_tokens.extend(accepted_tokens)
        print(f"[State] Committed = {self.committed_tokens}\n")

    def rollback(self):
        print(f"[Rollback] Clearing speculative tokens: {self.speculative_tokens}")
        self.speculative_tokens = []
        print(f"[State] Speculative cleared\n")

    def get_state(self):
        return {
            "committed": self.committed_tokens,
            "speculative": self.speculative_tokens
        }


# =========================
# DEMO RUN
# =========================

def run_demo():
    print("=== KV-CACHE DEMO START ===\n")

    cache = KVCache()

    # Step 1: Draft predicts tokens
    draft_tokens = ["A", "B", "C"]
    cache.add_speculative(draft_tokens)

    # Step 2: Target verifies
    target_tokens = ["A", "B", "X"]
    accepted = cache.verify(target_tokens)

    # Step 3: Commit accepted tokens
    cache.commit(accepted)

    # Step 4: Rollback remaining speculative tokens
    cache.rollback()

    # Final state
    final_state = cache.get_state()
    print(f"[Final State] {final_state}")
    print("\n=== KV-CACHE DEMO END ===")


# =========================
# TEST
# =========================

def test_kv_cache():
    cache = KVCache()

    draft_tokens = ["A", "B", "C"]
    cache.add_speculative(draft_tokens)

    target_tokens = ["A", "B", "X"]
    verified_tokens = cache.verify(target_tokens)

    cache.commit(verified_tokens)
    cache.rollback()

    state = cache.get_state()

    assert state["committed"] == ["A", "B"]
    assert state["speculative"] == []

    print("\n[TEST] KV-cache correctness test passed.")


if __name__ == "__main__":
    run_demo()
    test_kv_cache()

