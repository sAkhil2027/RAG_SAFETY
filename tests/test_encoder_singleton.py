import os
from encoder import get_encoder

def test_encoder_is_singleton():
    # Ensure environment variable does not affect singleton behavior
    os.environ.pop('EMBEDDING_MODEL_NAME', None)
    enc1 = get_encoder()
    enc2 = get_encoder()
    assert enc1 is enc2, "Encoder instances are not the same (singleton failed)"
