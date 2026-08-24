"""A multi tenant SaaS: composite keys, a tenant drop, and the bill for each layout.

Run with: python -m examples.tenants
"""

from __future__ import annotations

import random

from store.composite import encode, encode_field
from store.rangedel import Ranged
from store.shard import Modulo, Ranges, spread

TENANTS = 12
ROWS = 300


def load(store: Ranged, seed: int = 9) -> None:
    """Every tenant's rows, keyed tenant first so a tenant is a contiguous range."""
    source = random.Random(seed)
    for tenant in range(TENANTS):
        for _ in range(ROWS):
            key = encode((f"t{tenant:03d}".encode(), source.randbytes(6).hex().encode()))
            store.put(key, source.randbytes(16))


def drop_tenant(store: Ranged, tenant: int) -> None:
    """One range delete covers the whole tenant."""
    start = encode_field(f"t{tenant:03d}".encode())
    stop = encode_field(f"t{tenant + 1:03d}".encode())
    store.delete_range(start, stop)


def main() -> int:
    store = Ranged()
    load(store)
    before = len(store.keys())
    drop_tenant(store, 4)
    after = len(store.keys())
    print(f"rows before {before}, after dropping tenant 4: {after}")
    print(f"the drop wrote {store.range_writes} record(s) against {ROWS} rows")

    keys = [
        encode((f"t{at % TENANTS:03d}".encode(), f"r{at:05d}".encode())) for at in range(2000)
    ]
    hashed = Modulo(shards=6)
    ranged = Ranges(boundaries=[encode_field(f"t{at:03d}".encode()) for at in (2, 4, 6, 8, 10)])
    print(f"hash spread {spread(hashed, keys)}, range spread {spread(ranged, keys)}")
    tenant_keys = [key for key in keys if key.startswith(encode_field(b"t003"))]
    hash_shards = {hashed.place(key) for key in tenant_keys}
    range_shards = {ranged.place(key) for key in tenant_keys}
    print(
        f"tenant 3 scan touches {len(hash_shards)} hash shards, {len(range_shards)} range shard"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
