from typing import cast

from topobathykit.utils.cache import MemoizeWithLocks, concurrent_lru_cache


class MockResource:
    def __init__(self, val: int) -> None:
        self.val = val
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_memoize_lru_eviction() -> None:
    """
    Verify that the implementation of MemoizeWithLocks supports LRU eviction.
    """
    # Use a small maxsize
    cache_decorator = concurrent_lru_cache(maxsize=10)

    # Access the underlying cache instance
    memoizer = cast(MemoizeWithLocks, cache_decorator)

    resources: list[MockResource] = []

    @memoizer
    def dummy_func(x: int) -> MockResource:
        res = MockResource(x)
        resources.append(res)
        return res

    # Add 20 items (limit 10)
    for i in range(20):
        dummy_func(i)

    # Check cache size is capped at 10
    assert len(memoizer.cache) == 10, f"Cache size should be 10, but is {len(memoizer.cache)}"

    # Eviction must NOT close the evicted resource — other threads may hold a live reference
    # to it and be mid-read (e.g. a zarr store). Python's reference counting will close the
    # underlying file handle naturally once the last reference drops.
    for i in range(10):
        assert resources[i].closed is False, f"Resource {i} must not be closed on eviction"

    for i in range(10, 20):
        assert resources[i].closed is False, f"Resource {i} should still be open"

    print("LRU eviction does not close evicted resources — verified.")
