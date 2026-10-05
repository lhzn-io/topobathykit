import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MemoizeWithLocks:
    """
    A thread-safe LRU cache dictionary that ensures only one computation runs
    for a given input key, preventing "cache stampedes" where multiple threads
    fetch the same network resource concurrently.
    """

    def __init__(self, ttl: int | None = None, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self.cache: OrderedDict[Hashable, Any] = OrderedDict()
        self.locks: dict[Hashable, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a simple hashable key. Warning: Not perfect for nested dicts.
            try:
                # Discard 'cls' or 'self' if they are the first arguments and not useful for hashing
                # For safety, we just stringify the arguments.
                key = str(args) + str(frozenset(kwargs.items()))
            except Exception:
                logger.warning("MemoizeWithLocks failed to hash args, skipping cache.")
                return func(*args, **kwargs)

            # 1. Fast check
            with self._global_lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return cast(T, self.cache[key])
                if key not in self.locks:
                    self.locks[key] = threading.Lock()
                lock = self.locks[key]

            # 2. Acquire specific lock and compute if still missing
            with lock:
                # Double check under global lock
                with self._global_lock:
                    if key in self.cache:
                        self.cache.move_to_end(key)
                        return cast(T, self.cache[key])

                # Compute
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    # Don't cache failures
                    raise

                # Cache and release lock
                with self._global_lock:
                    self.cache[key] = result
                    self.cache.move_to_end(key)

                    # Eviction logic
                    while len(self.cache) > self.maxsize:
                        _, evicted_val = self.cache.popitem(last=False)
                        # Do NOT call .close() here. Other threads may hold a reference to
                        # the evicted value and be actively reading from it (e.g. a zarr-backed
                        # xarray DataArray mid-.load()). Calling .close() while another thread
                        # is reading closes the underlying zarr store, causing that thread's
                        # windowed read to fail with a closed-store error — the whole ncei_bag
                        # provider step then raises ProviderNoDataError and the tile falls back
                        # to BlueTopo. Python's reference counting will close file handles
                        # naturally once the last reference drops.
                        del evicted_val

                # We could delete the lock here, but keeping it is fine for memory until cache clears
            return result

        return wrapper


def concurrent_lru_cache(maxsize: int = 128) -> Callable[..., Any]:
    """
    Decorator for preventing cache stampedes in multithreaded fastAPI calls.
    Enforces an LRU eviction policy with size limit.
    """
    # Note: we use an instance per function, not a single global one
    decorator_instance = MemoizeWithLocks(maxsize=maxsize)
    return decorator_instance
