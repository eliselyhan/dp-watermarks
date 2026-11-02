"""Keyed token hashing and favored-color assignment."""
import hashlib
import struct
from decimal import Decimal, localcontext
import numpy as np


def prepare(sequences):
    """Encode each unique token sequence once; sampling still retains duplicates."""
    return {t: struct.pack('>' + 'q'*(len(t)+1), len(t), *t) for t in set(sequences)}


def hashes(prepared, key):
    template = hashlib.blake2b(digest_size=16, person=b'MarkLLM-RGS-v1',
                              key=hashlib.sha256(repr(key).encode()).digest())
    template.update(b'color\0')
    result = {}
    for tokens, payload in prepared.items():
        h = template.copy()
        h.update(payload)
        result[tokens] = int.from_bytes(h.digest(), 'big')
    return result


def colors(sequences, cache, p0):
    """Map keyed hashes to favored bits using the nominal probability p0."""
    with localcontext() as ctx:
        ctx.prec = 80
        threshold = int(Decimal(str(float(p0))) * Decimal(2**128))
    return np.array([cache[t] < threshold for t in sequences], dtype=np.int8)
