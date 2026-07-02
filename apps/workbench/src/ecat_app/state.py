"""Server-side session storage for live eCAT objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from .adapters import summarize_objects


@dataclass
class DatasetRecord:
    objects: list[object]
    warnings: list[str] = field(default_factory=list)


class SessionRegistry:
    """Keep live eCAT objects out of Dash JSON stores."""

    def __init__(self):
        self._datasets: dict[str, DatasetRecord] = {}

    def put(self, objects, warnings=None) -> str:
        dataset_id = uuid.uuid4().hex
        self._datasets[dataset_id] = DatasetRecord(list(objects or []), list(warnings or []))
        return dataset_id

    def get(self, dataset_id: str | None) -> list[object]:
        if not dataset_id or dataset_id not in self._datasets:
            return []
        return self._datasets[dataset_id].objects

    def warnings(self, dataset_id: str | None) -> list[str]:
        if not dataset_id or dataset_id not in self._datasets:
            return []
        return list(self._datasets[dataset_id].warnings)

    def snapshot(self, dataset_id: str | None) -> dict[str, object]:
        objects = self.get(dataset_id)
        return {
            "dataset_id": dataset_id,
            "summary": summarize_objects(objects),
            "warnings": self.warnings(dataset_id),
        }

    def get_included(self, dataset_id: str | None, row_ids: list[str] | None) -> list[object]:
        objects = self.get(dataset_id)
        if not row_ids:
            return objects
        included_indices = {
            int(str(row_id).replace("row-", ""))
            for row_id in row_ids
            if str(row_id).startswith("row-")
        }
        return [obj for index, obj in enumerate(objects) if index in included_indices]

    def get_by_row_ids(self, dataset_id: str | None, row_ids: list[str] | None) -> list[object]:
        objects = self.get(dataset_id)
        if not row_ids:
            return objects
        ordered = []
        for row_id in row_ids:
            if not str(row_id).startswith("row-"):
                continue
            index = int(str(row_id).replace("row-", ""))
            if 0 <= index < len(objects):
                ordered.append(objects[index])
        return ordered


registry = SessionRegistry()
