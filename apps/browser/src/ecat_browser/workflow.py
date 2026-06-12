"""Structured workflow state for the eCAT browser app."""

from dataclasses import dataclass, field


@dataclass
class BrowserWorkflow:
    app_mode: str = "local"
    source_kind: str | None = None
    source_path: str | None = None
    recursive: bool = False
    import_options: dict[str, object] = field(default_factory=dict)
    reference_settings: dict[str, object] = field(default_factory=dict)
    included_row_ids: list[str] = field(default_factory=list)
    selected_index: int | None = None
    filters: dict[str, object] = field(default_factory=dict)
    group_keys: list[str] = field(default_factory=list)
    sort_keys: list[str] = field(default_factory=list)
    analyses: list[str] = field(default_factory=list)
    plot_options: dict[str, object] = field(default_factory=dict)
    export_filename: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "app_mode": self.app_mode,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "recursive": self.recursive,
            "import_options": dict(self.import_options),
            "reference_settings": dict(self.reference_settings),
            "included_row_ids": list(self.included_row_ids),
            "selected_index": self.selected_index,
            "filters": dict(self.filters),
            "group_keys": list(self.group_keys),
            "sort_keys": list(self.sort_keys),
            "analyses": list(self.analyses),
            "plot_options": dict(self.plot_options),
            "export_filename": self.export_filename,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "BrowserWorkflow":
        if not data:
            return cls()
        return cls(
            app_mode=str(data.get("app_mode") or "local"),
            source_kind=data.get("source_kind"),
            source_path=data.get("source_path"),
            recursive=bool(data.get("recursive", False)),
            import_options=dict(data.get("import_options") or {}),
            reference_settings=dict(data.get("reference_settings") or {}),
            included_row_ids=list(data.get("included_row_ids") or []),
            selected_index=data.get("selected_index"),
            filters=dict(data.get("filters") or {}),
            group_keys=list(data.get("group_keys") or []),
            sort_keys=list(data.get("sort_keys") or []),
            analyses=list(data.get("analyses") or []),
            plot_options=dict(data.get("plot_options") or {}),
            export_filename=str(data.get("export_filename") or ""),
        )
