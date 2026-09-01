"""MySQL persistence and Laravel evidence-storage integration for Epic 5."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional

import mysql.connector
from mysql.connector import Error as MySQLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LARAVEL_ROOT = PROJECT_ROOT / "app"


def read_env(path: Path) -> Dict[str, str]:
    """Read the simple key/value entries used by Laravel's .env file."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class ViolationDBWriter:
    """Upsert violation events and publish their images to Laravel storage."""

    COLUMNS = (
        "event_id", "track_id", "plate_number", "speed", "speed_limit",
        "violation_type", "signal_state", "direction", "vehicle_color",
        "color_confidence", "frame_number", "frame_timestamp", "image_path",
        "plate_crop_path", "ocr_raw_text", "ocr_confidence", "ocr_engine",
    )

    def __init__(self, laravel_root: Path | str = LARAVEL_ROOT,
                 retry_attempts: int = 3, retry_delay: float = 0.5):
        self.laravel_root = Path(laravel_root).resolve()
        self.project_root = self.laravel_root.parent
        self.public_disk = self.laravel_root / "storage" / "app" / "public"
        self.evidence_dir = self.public_disk / "violations"
        self.pending_path = PROJECT_ROOT / "epic5" / "pending_db_events.json"
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay = max(0.0, retry_delay)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        env = read_env(self.laravel_root / ".env")
        self.config = {
            "host": os.getenv("DB_HOST", env.get("DB_HOST", "127.0.0.1")),
            "port": int(os.getenv("DB_PORT", env.get("DB_PORT", "3306"))),
            "database": os.getenv("DB_DATABASE", env.get("DB_DATABASE", "tvs")),
            "user": os.getenv("DB_USERNAME", env.get("DB_USERNAME", "root")),
            "password": os.getenv("DB_PASSWORD", env.get("DB_PASSWORD", "")),
            "charset": "utf8mb4",
            "use_unicode": True,
            "connection_timeout": 5,
        }

    @staticmethod
    def _event_dict(event: Any) -> Dict[str, Any]:
        if isinstance(event, dict):
            return dict(event)
        if hasattr(event, "to_dict"):
            return dict(event.to_dict())
        if is_dataclass(event):
            return asdict(event)
        raise TypeError("event must be a dict, dataclass, or expose to_dict()")

    def _source_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _publish_image(self, source_value: Optional[str], event_id: str,
                       filename: str) -> Optional[str]:
        source = self._source_path(source_value)
        if source is None or not source.is_file():
            return None
        destination_dir = self.evidence_dir / event_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename
        shutil.copy2(source, destination)
        # Laravel's Storage::url() expects a path relative to its public disk.
        return destination.relative_to(self.public_disk).as_posix()

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        return datetime.now()

    def prepare(self, event: Any) -> Dict[str, Any]:
        payload = self._event_dict(event)
        event_id = str(payload["event_id"])
        payload["image_path"] = self._publish_image(
            payload.get("image_path"), event_id, "violation.jpg"
        ) or payload.get("image_path")
        payload["plate_crop_path"] = self._publish_image(
            payload.get("plate_crop_path"), event_id, "plate.jpg"
        ) or payload.get("plate_crop_path")
        payload["speed_limit"] = payload.get("speed_limit_kmh",
                                              payload.get("speed_limit"))
        payload["speed"] = payload.get("speed_kmh", payload.get("speed"))
        payload["frame_timestamp"] = self._timestamp(
            payload.get("timestamp", payload.get("frame_timestamp"))
        )
        violation_type = payload.get("violation_type")
        payload["violation_type"] = getattr(violation_type, "value", violation_type)
        payload["plate_number"] = payload.get("plate_number") or "UNREADABLE"
        payload["vehicle_color"] = payload.get("vehicle_color") or "UNKNOWN"
        payload["color_confidence"] = payload.get("color_confidence") or 0.0
        return {column: payload.get(column) for column in self.COLUMNS}

    def _upsert(self, payload: Dict[str, Any]) -> None:
        placeholders = ", ".join(["%s"] * len(self.COLUMNS))
        columns = ", ".join(f"`{column}`" for column in self.COLUMNS)
        updates = ", ".join(
            f"`{column}` = VALUES(`{column}`)"
            for column in self.COLUMNS if column != "event_id"
        )
        sql = (
            f"INSERT INTO `violations` ({columns}, `created_at`, `updated_at`) "
            f"VALUES ({placeholders}, NOW(3), NOW(3)) "
            f"ON DUPLICATE KEY UPDATE {updates}, `updated_at` = NOW(3)"
        )
        connection = mysql.connector.connect(**self.config)
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, tuple(payload[column] for column in self.COLUMNS))
                connection.commit()
            finally:
                cursor.close()
        finally:
            connection.close()

    def _write_with_retry(self, payload: Dict[str, Any]) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._upsert(payload)
                return
            except MySQLError as exc:
                last_error = exc
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)
        assert last_error is not None
        raise last_error

    def _load_pending(self) -> Dict[str, Dict[str, Any]]:
        if not self.pending_path.exists():
            return {}
        try:
            return json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_pending(self, pending: Dict[str, Dict[str, Any]]) -> None:
        serializable = {
            key: {
                field: value.isoformat() if isinstance(value, datetime) else value
                for field, value in payload.items()
            }
            for key, payload in pending.items()
        }
        if not serializable:
            self.pending_path.unlink(missing_ok=True)
            return
        temp = self.pending_path.with_suffix(".tmp")
        temp.write_text(json.dumps(serializable, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        temp.replace(self.pending_path)

    def write_event(self, event: Any) -> bool:
        """Write immediately; spool the latest event state if MySQL is offline."""
        payload = self.prepare(event)
        try:
            self._write_with_retry(payload)
            return True
        except MySQLError as exc:
            pending = self._load_pending()
            pending[str(payload["event_id"])] = payload
            self._save_pending(pending)
            print(f"  [DB] MySQL unavailable; event queued: {exc}")
            return False

    def flush_pending(self) -> int:
        """Retry queued records and return the number successfully written."""
        pending = self._load_pending()
        written = 0
        for event_id, raw_payload in list(pending.items()):
            payload = dict(raw_payload)
            payload["frame_timestamp"] = self._timestamp(
                payload.get("frame_timestamp")
            )
            try:
                self._write_with_retry(payload)
            except MySQLError:
                continue
            pending.pop(event_id, None)
            written += 1
        self._save_pending(pending)
        return written

    def ping(self) -> bool:
        try:
            connection = mysql.connector.connect(**self.config)
            connection.close()
            return True
        except MySQLError:
            return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Epic 5 violation DB writer")
    parser.add_argument("--ping", action="store_true", help="test MySQL connection")
    parser.add_argument("--import-json", type=Path,
                        help="import events from a violation_events.json report")
    args = parser.parse_args()

    writer = ViolationDBWriter()
    if args.ping:
        connected = writer.ping()
        print("CONNECTED" if connected else "OFFLINE")
        return 0 if connected else 1
    if args.import_json:
        report = json.loads(args.import_json.read_text(encoding="utf-8"))
        events = report.get("rule_engine", {}).get("events", [])
        written = sum(writer.write_event(event) for event in events)
        print(f"Imported {written}/{len(events)} event(s)")
        return 0 if written == len(events) else 1
    flushed = writer.flush_pending()
    print(f"Flushed {flushed} queued event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
