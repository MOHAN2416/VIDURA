"""VIDURA Cloud Usage Accounting and Cost Tracking.

Tracks session and persistent cloud usage, enforces request and developer limits,
calculates configured/estimated costs without fabricating data, and supports retention cleanup.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.routing import classify_cloud_failure, FailureCategory

logger = logging.getLogger("VIDURA.models.usage")


@dataclass
class CloudUsageRecord:
    """Structured record of a single cloud inference attempt or fallback."""
    request_id: str
    timestamp: str
    requested_provider: str
    actual_provider: str
    model: str
    category: str
    success: bool
    fallback_used: bool = False
    fallback_reason: str | None = None
    failure_category: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float | None = None
    estimated_cost: float | None = None
    cost_status: str = "not_configured"

    def to_dict(self) -> dict[str, Any]:
        """Converts usage record to a dictionary without sensitive data."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            "model": self.model,
            "category": self.category,
            "success": self.success,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "failure_category": self.failure_category,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "estimated_cost": self.estimated_cost,
            "cost_status": self.cost_status,
        }


class CloudUsageTracker:
    """Application-level tracker for cloud inferences, budget limits, and usage persistence."""

    def __init__(
        self,
        config: Any = None,
        max_requests: int | None = None,
        max_developer_requests: int | None = None,
        input_token_price: float | None = None,
        output_token_price: float | None = None,
        db_path: Path | str | None = None,
        retention_days: int | None = None,
    ) -> None:
        if config is not None:
            if max_requests is None:
                max_requests = getattr(config, "vidura_cloud_max_requests", None)
            if max_developer_requests is None:
                max_developer_requests = getattr(config, "vidura_cloud_max_developer_requests", None)
            if input_token_price is None:
                input_token_price = getattr(config, "vidura_cloud_input_token_price", None)
            if output_token_price is None:
                output_token_price = getattr(config, "vidura_cloud_output_token_price", None)
            if retention_days is None:
                retention_days = getattr(config, "vidura_cloud_usage_retention_days", 30)
            if db_path is None:
                db_path = getattr(config, "db_path", None)

        self.max_requests = max_requests if (max_requests is not None and max_requests > 0) else None
        self.max_developer_requests = (
            max_developer_requests if (max_developer_requests is not None and max_developer_requests > 0) else None
        )
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price
        self.retention_days = int(retention_days) if retention_days is not None else 30
        self.db_path = Path(db_path).resolve() if db_path else None

        # In-memory session accounting
        self._session_cloud_attempts: int = 0
        self._session_cloud_success: int = 0
        self._session_cloud_failed: int = 0
        self._session_fallbacks: int = 0
        self._session_dev_cloud_requests: int = 0
        self._session_local_requests: int = 0
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_total_tokens: int = 0

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection | None:
        if not self.db_path:
            return None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as err:
            logger.warning(f"Failed to connect to SQLite usage database: {err}")
            return None

    def _init_db(self) -> None:
        """Initializes SQLite schema for cloud usage records."""
        conn = self._get_connection()
        if not conn:
            return
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cloud_usage_records (
                        request_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        requested_provider TEXT NOT NULL,
                        actual_provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        category TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        fallback_used INTEGER NOT NULL,
                        fallback_reason TEXT,
                        failure_category TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        duration_seconds REAL,
                        estimated_cost REAL,
                        cost_status TEXT NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_usage_time ON cloud_usage_records(timestamp);")
        except Exception as err:
            logger.warning(f"Failed to initialize cloud usage table: {err}")
        finally:
            conn.close()

    @property
    def cloud_attempts(self) -> int:
        return self._session_cloud_attempts

    @property
    def cloud_successes(self) -> int:
        return self._session_cloud_success

    @property
    def cloud_failures(self) -> int:
        return self._session_cloud_failed

    @property
    def fallbacks(self) -> int:
        return self._session_fallbacks

    @property
    def developer_cloud_requests(self) -> int:
        return self._session_dev_cloud_requests

    @property
    def local_requests(self) -> int:
        return self._session_local_requests

    def is_cloud_limit_reached(self) -> tuple[bool, str]:
        """Checks if session-level cloud request limit has been reached."""
        if self.max_requests is not None and self._session_cloud_attempts >= self.max_requests:
            return True, f"Session cloud request limit reached ({self._session_cloud_attempts}/{self.max_requests})"
        return False, ""

    def is_developer_cloud_limit_reached(self) -> tuple[bool, str]:
        """Checks if developer cloud request limit has been reached."""
        if self.max_developer_requests is not None and self._session_dev_cloud_requests >= self.max_developer_requests:
            return True, f"Developer cloud request limit reached ({self._session_dev_cloud_requests}/{self.max_developer_requests})"
        return False, ""

    def check_limits(self, is_developer: bool = False) -> tuple[bool, str]:
        """Evaluates all cloud limits prior to sending a network request."""
        reached, reason = self.is_cloud_limit_reached()
        if reached:
            return True, reason
        if is_developer:
            dev_reached, dev_reason = self.is_developer_cloud_limit_reached()
            if dev_reached:
                return True, dev_reason
        return False, ""

    def check_can_request(self, is_developer: bool = False) -> tuple[bool, str]:
        """Returns True if a request is permitted under current limits, else False and reason."""
        reached, reason = self.check_limits(is_developer=is_developer)
        return not reached, reason

    def record_cloud_attempt(self, is_developer: bool = False) -> None:
        """Records that a cloud inference attempt is actively being sent."""
        self._session_cloud_attempts += 1
        if is_developer:
            self._session_dev_cloud_requests += 1

    def _calculate_cost(self, input_tokens: int | None, output_tokens: int | None) -> tuple[float | None, str]:
        """Calculates estimated cost if pricing is explicitly configured."""
        if self.input_token_price is not None and self.output_token_price is not None:
            in_t = input_tokens or 0
            out_t = output_tokens or 0
            cost = (in_t / 1000.0 * self.input_token_price) + (out_t / 1000.0 * self.output_token_price)
            return round(cost, 6), "estimated_configured"
        return None, "not_configured"

    def record_cloud_success(
        self,
        request_id: str,
        model: str,
        category: str = "general",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_seconds: float | None = None,
        is_developer: bool = False,
        task_type: str | None = None,
        complexity: str | None = None,
        fallback_used: bool = False,
        **kwargs: Any,
    ) -> CloudUsageRecord:
        """Records a successful cloud response and persists the record."""
        self._session_cloud_success += 1
        if input_tokens is not None:
            self._session_input_tokens += input_tokens
        if output_tokens is not None:
            self._session_output_tokens += output_tokens
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
            self._session_total_tokens += total_tokens

        cost, cost_status = self._calculate_cost(input_tokens, output_tokens)
        timestamp = datetime.now(timezone.utc).isoformat()

        cat = task_type or category or ("developer" if is_developer else "general")
        record = CloudUsageRecord(
            request_id=request_id,
            timestamp=timestamp,
            requested_provider="cloud",
            actual_provider="cloud",
            model=model,
            category=cat,
            success=True,
            fallback_used=fallback_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            estimated_cost=cost,
            cost_status=cost_status,
        )
        self._persist_record(record)
        return record

    def record_cloud_failure(
        self,
        request_id: str,
        model: str,
        category: str = "general",
        error: Exception | None = None,
        error_message: str | None = None,
        is_developer: bool = False,
        task_type: str | None = None,
        complexity: str | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        actual_provider: str = "cloud",
        **kwargs: Any,
    ) -> CloudUsageRecord:
        """Records a failed cloud inference and optional local fallback."""
        self._session_cloud_failed += 1
        if fallback_used:
            self._session_fallbacks += 1

        if error is not None:
            fail_cat = classify_cloud_failure(error).value
        elif error_message is not None:
            fail_cat = FailureCategory.TRANSIENT.value
        else:
            fail_cat = FailureCategory.UNKNOWN.value

        timestamp = datetime.now(timezone.utc).isoformat()
        reason_str = str(fallback_reason or error or error_message or "")
        cat = task_type or category or ("developer" if is_developer else "general")

        record = CloudUsageRecord(
            request_id=request_id,
            timestamp=timestamp,
            requested_provider="cloud",
            actual_provider=actual_provider,
            model=model,
            category=cat,
            success=False,
            fallback_used=fallback_used,
            fallback_reason=reason_str if reason_str else None,
            failure_category=fail_cat,
            cost_status="not_configured" if not (self.input_token_price and self.output_token_price) else "estimated_configured",
        )
        self._persist_record(record)
        return record

    def record_fallback(self, request_id: str, reason: str | None = None) -> None:
        """Explicitly records that a fallback was engaged."""
        self._session_fallbacks += 1

    def record_local_request(self, category: str = "general") -> None:
        """Records a purely local inference (never counted as cloud)."""
        self._session_local_requests += 1

    def _persist_record(self, record: CloudUsageRecord) -> None:
        conn = self._get_connection()
        if not conn:
            return
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO cloud_usage_records (
                        request_id, timestamp, requested_provider, actual_provider,
                        model, category, success, fallback_used, fallback_reason,
                        failure_category, input_tokens, output_tokens, total_tokens,
                        duration_seconds, estimated_cost, cost_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        record.request_id,
                        record.timestamp,
                        record.requested_provider,
                        record.actual_provider,
                        record.model,
                        record.category,
                        1 if record.success else 0,
                        1 if record.fallback_used else 0,
                        record.fallback_reason,
                        record.failure_category,
                        record.input_tokens,
                        record.output_tokens,
                        record.total_tokens,
                        record.duration_seconds,
                        record.estimated_cost,
                        record.cost_status,
                    ),
                )
        except Exception as err:
            logger.warning(f"Failed to persist cloud usage record: {err}")
        finally:
            conn.close()

    def prune_retention(self, retention_days: int | None = None, days: int | None = None) -> int:
        """Prunes persistent usage records older than retention_days."""
        target_days = days if days is not None else (retention_days if retention_days is not None else self.retention_days)
        conn = self._get_connection()
        if not conn:
            return 0
        deleted_count = 0
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM cloud_usage_records WHERE datetime(timestamp) < datetime('now', ?);",
                    (f"-{target_days} days",),
                )
                deleted_count = cursor.rowcount
            logger.info(f"Pruned {deleted_count} cloud usage records older than {target_days} days.")
        except Exception as err:
            logger.warning(f"Failed to prune retention records: {err}")
        finally:
            conn.close()
        return deleted_count


    def get_summary(self) -> dict[str, Any]:
        """Returns safe, aggregated session metrics with no secrets."""
        cost = None
        cost_status = "not_configured"
        if self.input_token_price is not None and self.output_token_price is not None:
            cost = round(
                (self._session_input_tokens / 1000.0 * self.input_token_price)
                + (self._session_output_tokens / 1000.0 * self.output_token_price),
                6,
            )
            cost_status = "estimated_configured"

        return {
            "cloud_requests_attempted": self._session_cloud_attempts,
            "cloud_requests_successful": self._session_cloud_success,
            "cloud_requests_failed": self._session_cloud_failed,
            "fallback_requests": self._session_fallbacks,
            "developer_cloud_requests": self._session_dev_cloud_requests,
            "local_requests": self._session_local_requests,
            "input_tokens": self._session_input_tokens if self._session_input_tokens > 0 else None,
            "output_tokens": self._session_output_tokens if self._session_output_tokens > 0 else None,
            "total_tokens": self._session_total_tokens if self._session_total_tokens > 0 else None,
            "estimated_cost": cost,
            "cost_status": cost_status,
            "max_requests": self.max_requests,
            "max_developer_requests": self.max_developer_requests,
        }


def estimate_tokens(content: Any) -> int:
    """Estimates the number of tokens in the given content using character-based approximation.

    Approximately 4 characters per token for typical code and natural language text.
    Handles strings, lists of message dicts, and arbitrary text objects.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        text = " ".join(parts)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
