"""Prediction caching layer. Uses in-memory dict fallback if Redis unavailable."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PredictionCache:
    """Caching layer for prediction results.

    Attempts to connect to Redis for distributed caching. Falls back
    to a thread-safe in-memory dict with TTL if Redis is unavailable.

    Usage::

        cache = PredictionCache(redis_url="redis://localhost:6379")
        cache.set("predict:pid:012-345-678", result_dict, ttl_hours=24)
        cached = cache.get("predict:pid:012-345-678")

    Args:
        redis_url: Redis connection URL (e.g. "redis://localhost:6379").
            If None or connection fails, falls back to in-memory caching.
        default_ttl_hours: Default time-to-live for cached entries in hours.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl_hours: int = 24,
    ) -> None:
        self.default_ttl_hours = default_ttl_hours
        self._redis = None
        self._use_redis = False

        # In-memory fallback: key -> (value, expiry_timestamp)
        self._memory_cache: dict[str, tuple[dict, float]] = {}

        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                # Test the connection
                self._redis.ping()
                self._use_redis = True
                logger.info(
                    "PredictionCache connected to Redis at %s", redis_url,
                )
            except Exception as exc:
                logger.warning(
                    "Redis connection failed (%s); falling back to in-memory cache",
                    exc,
                )
                self._redis = None
                self._use_redis = False
        else:
            logger.info(
                "PredictionCache using in-memory cache (no Redis URL provided)",
            )

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a cached prediction result.

        Args:
            key: Cache key string.

        Returns:
            Cached dict or None if not found or expired.
        """
        if self._use_redis:
            return self._redis_get(key)
        return self._memory_get(key)

    def set(
        self,
        key: str,
        value: dict,
        ttl_hours: Optional[int] = None,
    ) -> None:
        """Store a prediction result in the cache.

        Args:
            key: Cache key string.
            value: Dict to cache (must be JSON-serializable).
            ttl_hours: Time-to-live in hours. Uses default if not specified.
        """
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours

        if self._use_redis:
            self._redis_set(key, value, ttl)
        else:
            self._memory_set(key, value, ttl)

    def invalidate(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.

        Useful for clearing predictions for a retrained segment.

        Args:
            pattern: Key pattern to match. For Redis, uses SCAN with
                glob-style patterns. For in-memory, uses simple substring
                matching.

        Returns:
            Number of keys invalidated.
        """
        if self._use_redis:
            return self._redis_invalidate(pattern)
        return self._memory_invalidate(pattern)

    def clear(self) -> None:
        """Clear all cached predictions.

        For Redis, flushes the current database. For in-memory, empties
        the dict.
        """
        if self._use_redis:
            try:
                self._redis.flushdb()
                logger.info("Redis cache cleared")
            except Exception as exc:
                logger.warning("Failed to clear Redis cache: %s", exc)
        else:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            logger.info("In-memory cache cleared (%d entries)", count)

    # ================================================================
    # CACHE KEY HELPERS
    # ================================================================

    @staticmethod
    def make_prediction_key(
        pid: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        address: Optional[str] = None,
        property_type: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> str:
        """Generate a deterministic cache key for a prediction request.

        The key is a SHA-256 hash of the normalized request parameters.
        Address is included because different units at the same lat/lon
        (e.g. #101 vs #302 in a condo building) must resolve to
        different properties.

        Args:
            pid: Property PID.
            lat: Latitude.
            lon: Longitude.
            address: Street address including unit number.
            property_type: Property type string.
            overrides: Feature overrides dict.

        Returns:
            Cache key string prefixed with "predict:".
        """
        key_parts = {
            "pid": pid,
            "lat": round(lat, 6) if lat is not None else None,
            "lon": round(lon, 6) if lon is not None else None,
            "address": address,
            "property_type": property_type,
            "overrides": overrides,
        }

        key_json = json.dumps(key_parts, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]

        return f"predict:{key_hash}"

    # ================================================================
    # REDIS IMPLEMENTATION
    # ================================================================

    def _redis_get(self, key: str) -> Optional[dict]:
        """Get value from Redis."""
        try:
            raw = self._redis.get(key)
            if raw is not None:
                logger.debug("Redis cache hit: %s", key)
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis GET failed for key '%s': %s", key, exc)
        return None

    def _redis_set(self, key: str, value: dict, ttl_hours: int) -> None:
        """Set value in Redis with TTL."""
        try:
            ttl_seconds = ttl_hours * 3600
            serialized = json.dumps(value, default=str)
            self._redis.setex(key, ttl_seconds, serialized)
            logger.debug("Redis cache set: %s (TTL=%dh)", key, ttl_hours)
        except Exception as exc:
            logger.warning("Redis SET failed for key '%s': %s", key, exc)

    def _redis_invalidate(self, pattern: str) -> int:
        """Invalidate keys in Redis matching a pattern."""
        try:
            count = 0
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor, match=pattern, count=100,
                )
                if keys:
                    self._redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break

            logger.info(
                "Redis cache invalidated %d keys matching '%s'", count, pattern,
            )
            return count
        except Exception as exc:
            logger.warning("Redis invalidation failed for '%s': %s", pattern, exc)
            return 0

    # ================================================================
    # IN-MEMORY IMPLEMENTATION
    # ================================================================

    def _memory_get(self, key: str) -> Optional[dict]:
        """Get value from in-memory cache, respecting TTL."""
        entry = self._memory_cache.get(key)
        if entry is None:
            return None

        value, expiry = entry
        if time.time() > expiry:
            # Entry has expired
            del self._memory_cache[key]
            logger.debug("Memory cache expired: %s", key)
            return None

        logger.debug("Memory cache hit: %s", key)
        return value

    def _memory_set(self, key: str, value: dict, ttl_hours: int) -> None:
        """Set value in in-memory cache with TTL."""
        expiry = time.time() + (ttl_hours * 3600)
        self._memory_cache[key] = (value, expiry)
        logger.debug("Memory cache set: %s (TTL=%dh)", key, ttl_hours)

        # Lazy cleanup: remove expired entries if cache is getting large
        if len(self._memory_cache) > 10_000:
            self._memory_cleanup()

    def _memory_invalidate(self, pattern: str) -> int:
        """Invalidate keys in in-memory cache matching a substring pattern."""
        # Convert glob-style pattern to simple substring match
        search_term = pattern.replace("*", "")
        keys_to_delete = [
            k for k in self._memory_cache if search_term in k
        ]
        for k in keys_to_delete:
            del self._memory_cache[k]

        logger.info(
            "Memory cache invalidated %d keys matching '%s'",
            len(keys_to_delete),
            pattern,
        )
        return len(keys_to_delete)

    def _memory_cleanup(self) -> None:
        """Remove all expired entries from the in-memory cache."""
        now = time.time()
        expired = [
            k for k, (_, expiry) in self._memory_cache.items() if now > expiry
        ]
        for k in expired:
            del self._memory_cache[k]

        if expired:
            logger.info(
                "Memory cache cleanup: removed %d expired entries", len(expired),
            )
