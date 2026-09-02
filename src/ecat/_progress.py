"""Shared notebook/terminal progress helpers."""

from __future__ import annotations

import html
import time


def progress_enabled(progress):
    if progress is True:
        return True
    if progress in (False, None):
        return False
    if callable(progress):
        return True
    if isinstance(progress, str):
        return progress.strip().lower() not in {"", "false", "none", "off", "disable", "disabled"}
    return bool(progress)


def format_progress_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        if seconds >= 10:
            return f"{seconds:.0f}s"
        if seconds >= 1:
            return f"{seconds:.1f}s"
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remainder = int(round(seconds - 60 * minutes))
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes:02d}m"


class NotebookProgressDisplay:
    def __init__(
        self,
        *,
        total=None,
        label="Progress",
        leave=True,
        unit="items",
        approx_total=False,
        metric_label=None,
        indeterminate=None,
    ):
        from IPython.display import HTML, display

        self.HTML = HTML
        self.total = None if total is None else max(1, int(total))
        self.indeterminate = bool(indeterminate if indeterminate is not None else (total is None))
        self.label = label
        self.leave = leave
        self.unit = unit
        self.approx_total = bool(approx_total)
        self.metric_label = metric_label
        self.start_time = time.monotonic()
        self.count = 0
        self.metric = None
        self.handle = display(
            self.HTML(
                self._html(
                    0,
                    None,
                    self.total,
                    label,
                    elapsed=0.0,
                    remaining=None,
                    unit=unit,
                    indeterminate=self.indeterminate,
                    approx_total=self.approx_total,
                    metric_label=metric_label,
                )
            ),
            display_id=True,
        )
        if self.handle is None:
            raise RuntimeError("IPython display did not provide an updatable display handle.")

    def update(self, count, *, metric=None):
        self.count = int(count)
        self.metric = metric
        elapsed = max(0.0, time.monotonic() - self.start_time)
        remaining = self._estimate_remaining(count, elapsed)
        self.handle.update(
            self.HTML(
                self._html(
                    count,
                    metric,
                    self.total,
                    self.label,
                    elapsed=elapsed,
                    remaining=remaining,
                    unit=self.unit,
                    indeterminate=self.indeterminate,
                    approx_total=self.approx_total,
                    metric_label=self.metric_label,
                )
            )
        )

    def close(self):
        if not self.leave:
            self.handle.update(self.HTML(""))
            return
        elapsed = max(0.0, time.monotonic() - self.start_time)
        self.handle.update(
            self.HTML(
                self._done_html(
                    self.count,
                    self.metric,
                    self.total,
                    self.label,
                    elapsed=elapsed,
                    unit=self.unit,
                    indeterminate=self.indeterminate,
                    approx_total=self.approx_total,
                    metric_label=self.metric_label,
                )
            )
        )

    def _estimate_remaining(self, count, elapsed):
        if self.indeterminate:
            return None
        if self.total is None or int(count) <= 0 or int(count) >= int(self.total):
            return None
        remaining_units = max(0, int(self.total) - int(count))
        if remaining_units <= 0:
            return None
        return elapsed * remaining_units / max(1, int(count))

    @staticmethod
    def _count_text(count, total, *, unit="items", approx_total=False):
        count = int(count)
        if total is None:
            return f"{count} {unit}"
        total_text = f"~{int(total)}" if approx_total else f"{int(total)}"
        return f"{count} / {total_text} {unit}"

    @staticmethod
    def _metric_text(metric, *, metric_label=None):
        if metric is None:
            return ""
        if metric_label:
            return f"{metric_label} {metric:.3g}"
        return f"{metric:.3g}"

    @classmethod
    def _html(
        cls,
        count,
        metric,
        total,
        label,
        *,
        elapsed=0.0,
        remaining=None,
        unit="items",
        indeterminate=False,
        approx_total=False,
        metric_label=None,
    ):
        label = html.escape(str(label))
        count_text = cls._count_text(count, total, unit=unit, approx_total=approx_total)
        if indeterminate:
            top_right = count_text
            progress_html = (
                '<div style="height:10px; border-radius:999px; background:#e5e7eb; overflow:hidden; position:relative;">'
                '<div style="position:absolute; height:10px; width:35%; left:-35%; top:0; background:#2563eb; border-radius:999px; '
                'animation: ecat-indeterminate 1.2s linear infinite;"></div>'
                '</div>'
                '<style>@keyframes ecat-indeterminate { 0% { transform: translateX(0); } 100% { transform: translateX(300%); } }</style>'
            )
        else:
            total_int = max(1, int(total))
            pct = 100.0 * float(count) / float(total_int)
            top_right = f"{count_text} ({pct:.0f}%)"
            width_pct = min(max(pct, 0.0), 100.0)
            progress_html = (
                '<div style="height:10px; border-radius:999px; background:#e5e7eb; overflow:hidden;">'
                f'<div style="height:10px; width:{width_pct:.4f}%; background:#2563eb;"></div>'
                "</div>"
            )

        timing_parts = [f"elapsed {format_progress_duration(elapsed)}"]
        if remaining is not None and float(remaining) > 0:
            timing_parts.append(f"remaining ~{format_progress_duration(remaining)}")
        metric_text = cls._metric_text(metric, metric_label=metric_label)
        if metric_text:
            timing_parts.append(metric_text)
        timing_text = " | ".join(timing_parts)
        return f"""
        <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 520px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:13px;">
            <span>{label}</span>
            <span>{html.escape(top_right)}</span>
          </div>
          {progress_html}
          <div style="margin-top:4px; font-size:12px; color:#4b5563;">{html.escape(timing_text)}</div>
        </div>
        """

    @classmethod
    def _done_html(
        cls,
        count,
        metric,
        total,
        label,
        *,
        elapsed=0.0,
        unit="items",
        indeterminate=False,
        approx_total=False,
        metric_label=None,
    ):
        if indeterminate and total is None:
            label = html.escape(str(label))
            count_text = cls._count_text(count, total, unit=unit, approx_total=approx_total)
            progress_html = (
                '<div style="height:10px; border-radius:999px; background:#e5e7eb; overflow:hidden;">'
                '<div style="height:10px; width:100%; background:#2563eb;"></div>'
                "</div>"
            )
            timing_text = f"elapsed {format_progress_duration(elapsed)}"
            metric_text = cls._metric_text(metric, metric_label=metric_label)
            if metric_text:
                timing_text = f"{timing_text} | {metric_text}"
            return f"""
            <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 520px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:13px;">
                <span>{label}</span>
                <span>{html.escape(count_text)}</span>
              </div>
              {progress_html}
              <div style="margin-top:4px; font-size:12px; color:#4b5563;">{html.escape(timing_text)}</div>
            </div>
            """
        return cls._html(
            count,
            metric,
            total,
            label,
            elapsed=elapsed,
            remaining=None,
            unit=unit,
            indeterminate=indeterminate,
            approx_total=approx_total,
            metric_label=metric_label,
        )
