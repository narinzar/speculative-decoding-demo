"""Optional integration test that actually loads tiny HF models.

Skipped automatically when offline or when transformers is unavailable, so the
main test suite never triggers a large download. Uses "sshleifer/tiny-gpt2" for
both draft and target (a few hundred KB) to exercise the real interfaces.
"""
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _online(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    not _online(), reason="offline: skipping model-download integration test"
)


def test_load_tiny_pair_and_generate():
    import torch

    from src.models import full_logits, load_pair, next_token_logits
    from src.spec_decode import speculative_generate

    tiny = "sshleifer/tiny-gpt2"
    draft, target, tok = load_pair(tiny, tiny, device="cpu")

    ids = tok("hello world", return_tensors="pt").input_ids
    res = speculative_generate(
        ids,
        lambda x: next_token_logits(draft, x),
        lambda x: full_logits(target, x),
        max_new_tokens=8,
        k=3,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        generator=torch.Generator().manual_seed(0),
    )
    # draft == target so every proposal is accepted.
    assert res.accepted == res.proposed
    assert len(res.tokens) <= 8
