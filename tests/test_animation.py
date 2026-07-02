from types import MappingProxyType
from unittest.mock import ANY

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import pytest


def test_animate_returns_animation_result_for_single_cv(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.__class__.__name__ == "AnimationResult"
    assert hasattr(result, "animation")
    assert hasattr(result, "figure")
    assert hasattr(result, "axes")
    assert hasattr(result, "summary")
    assert result.figure is not None
    assert result.axes is not None


def test_animate_builds_static_single_plot_figure(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(cv_obj, {"plot": False, "title": False, "legend": False})

    assert result.figure is result.axes.figure
    assert len(result.axes.lines) == 1


def test_animate_single_object_default_does_not_force_legend(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.axes.get_legend() is None


def test_animate_builds_static_multiplot_figure(ecat_module, cv_factory):
    first = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="200mVs_sample_Ar_MeCN_10mM_Fc_run02")

    result = ecat_module.animate([first, second], {"plot": False, "title": False, "legend": False})

    assert result.figure is result.axes.figure
    assert len(result.axes.lines) == 2


def test_cv_object_animate_forwards_self_and_options(monkeypatch, cv_factory):
    import ecat.animation as animation_module

    cv_obj = cv_factory()
    options = {"plot": False}
    calls = {}

    def fake_animate(obj_or_list, forwarded_options=None):
        calls["obj_or_list"] = obj_or_list
        calls["options"] = forwarded_options
        return "sentinel-result"

    monkeypatch.setattr(animation_module, "animate", fake_animate)

    result = cv_obj.animate(options)

    assert result == "sentinel-result"
    assert calls == {"obj_or_list": cv_obj, "options": options}


def test_animation_result_save_rejects_unknown_formats(ecat_module, cv_factory, tmp_path):
    result = ecat_module.animate(cv_factory(), {"plot": False})

    with pytest.raises(ValueError, match="HTML, GIF, or MP4"):
        result.save(tmp_path / "placeholder.webp")


def test_animation_wraps_real_matplotlib_animation(ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})

    assert hasattr(result.animation, "to_jshtml")


def test_animate_normalizes_mapping_like_options(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(
        cv_obj,
        MappingProxyType({"plot": False, "animate_repeat": True}),
    )

    assert result.summary["options"] == {"plot": False, "animate repeat": True}


def test_animate_accepts_space_style_animation_options(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(
        cv_obj,
        {
            "plot": False,
            "trace mode": "draw",
            "timing mode": "normalized",
            "end hold": 1.5,
        },
    )

    assert result.summary["options"]["trace mode"] == "draw"
    assert result.summary["options"]["timing mode"] == "normalized"
    assert result.summary["options"]["end hold"] == 1.5


def test_animate_stride_thins_trace_payload_without_mutating_object(ecat_module, cv_factory):
    cv_obj = cv_factory()
    original_points = len(cv_obj.x({"one column": True}))

    result = ecat_module.animate(cv_obj, {"plot": False, "stride": 3})

    assert result.summary["stride"] == 3
    assert result.summary["resolved options"]["stride"] == 3
    assert len(result.axes.lines[0].get_xdata()) == 8
    assert len(cv_obj.x({"one column": True})) == original_points


def test_animate_rejects_non_positive_stride(ecat_module, cv_factory):
    with pytest.raises(ecat_module.OptionError, match="stride"):
        ecat_module.animate(cv_factory(), {"stride": 0})


def test_animate_accepts_multiplot_style_options_for_lists(ecat_module, cv_factory):
    first = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_CO2_MeCN_20mM_Fc_run02")

    result = ecat_module.animate(
        [first, second],
        {
            "plot": False,
            "plot labels": ["10 mM Fc", "20 mM Fc"],
            "legend outside": True,
        },
    )

    assert result.summary["options"]["labels"] == ["10 mM Fc", "20 mM Fc"]
    assert result.summary["resolved options"]["plot labels"] == ["10 mM Fc", "20 mM Fc"]
    assert result.summary["resolved options"]["legend outside"] is True


def test_animate_rejects_invalid_animation_option_value(ecat_module, cv_factory):
    cv_obj = cv_factory()

    with pytest.raises(ecat_module.OptionError, match="trace mode"):
        ecat_module.animate(cv_obj, {"trace mode": "banana"})


def test_animation_single_object_defaults_to_draw(ecat_module, cv_factory):
    cv_obj = cv_factory()

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.summary["trace mode"] == "draw"
    assert result.summary["schedule"] is None


def test_animation_list_with_mixed_scan_rates_defaults_to_draw_and_simultaneous(
    ecat_module,
    cv_factory,
):
    slow = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    fast = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")

    result = ecat_module.animate([slow, fast], {"plot": False})

    assert result.summary["trace mode"] == "draw"
    assert result.summary["schedule"] == "simultaneous"


def test_animation_list_with_uniform_scan_rates_defaults_to_instant_and_staggered(
    ecat_module,
    cv_factory,
):
    first = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")
    second = cv_factory(name="100mVs_sample_N2_DMF_5mM_Fc_run02")

    result = ecat_module.animate([first, second], {"plot": False})

    assert result.summary["trace mode"] == "instant"
    assert result.summary["schedule"] == "staggered"


def test_animation_uses_auto_timing_mode_when_metadata_supports_physical_timing(
    ecat_module,
    cv_factory,
):
    cv_obj = cv_factory()
    cv_obj.scan_rate = 0.25

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.summary["timing mode"] == "physical"


@pytest.mark.parametrize("scan_rate", [0, -0.25, float("nan")])
def test_animation_timing_mode_ignores_unusable_scan_rate_without_time_metadata(
    ecat_module,
    cv_factory,
    scan_rate,
):
    obj = cv_factory()
    obj.scan_rate = scan_rate

    result = ecat_module.animate(obj, {"plot": False})

    assert result.summary["timing mode"] == "normalized"


def test_animation_timing_mode_uses_physical_when_time_column_is_available(
    ecat_module,
    blank_echem_factory,
):
    obj = blank_echem_factory()
    obj.scan_rate = float("nan")
    obj.data = pd.DataFrame({"Time/sec": [0.0, 0.5, 1.0], "Current (A)": [0.0, 1.0, 0.5]})
    obj.num_x_cols = 1
    obj.compounds = []
    obj.concentrations = []

    result = ecat_module.animate(obj, {"plot": False})

    assert result.summary["timing mode"] == "physical"


def test_animation_falls_back_to_normalized_timing_when_physical_timing_is_unavailable(
    ecat_module,
    cv_factory,
):
    obj = cv_factory()
    obj.scan_rate = None

    result = ecat_module.animate(obj, {"plot": False})

    assert result.summary["timing mode"] == "normalized"


def test_animation_excludes_quiet_time_by_default(ecat_module, cv_factory):
    cv_obj = cv_factory()
    cv_obj.quiet_time = 2.0

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.summary["include quiet time"] is False


def test_animation_preserves_explicit_quiet_time_false_option(ecat_module, cv_factory):
    cv_obj = cv_factory()
    cv_obj.quiet_time = 2.0

    result = ecat_module.animate(cv_obj, {"plot": False, "include quiet time": False})

    assert result.summary["include quiet time"] is False


def test_animation_result_summary_reports_resolved_settings(ecat_module, cv_factory):
    cv_obj = cv_factory()
    cv_obj.scan_rate = 0.1

    result = ecat_module.animate(cv_obj, {"plot": False})

    assert result.summary["trace mode"] == "draw"
    assert result.summary["fps"] == 20
    assert result.summary["loop"] is True
    assert result.summary["end hold"] == 2


def test_animation_result_to_html_returns_string(ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})

    html = result.to_html()

    assert isinstance(html, str)
    assert html != ""


def test_animation_render_override_ignores_internal_underscore_keys(ecat_module, cv_factory):
    result = ecat_module.animate(
        cv_factory(),
        {"plot": False, "print": False, "plot labels": ["Trace A"]},
    )

    resolved = dict(result.summary["resolved options"])
    resolved["_deduplicate labels explicit"] = False

    html = result.to_html(options=resolved)

    assert isinstance(html, str)
    assert html != ""


def test_animation_autoplay_helper_injects_play_call(ecat_module):
    import ecat.animation as animation_module

    html = """
<script>
setTimeout(function() {
    animabc123 = new Animation(frames, img_id, slider_id, 50.0,
                         loop_select_id);
}, 0);
</script>
"""

    updated = animation_module._autoplay_html(html)

    assert "animabc123.play_animation();" in updated


def test_animation_result_show_returns_none(ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    result.to_html = lambda options=None: "<html>ok</html>"

    assert result.show() is None


def test_animation_result_display_returns_none(ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    result.to_html = lambda options=None: "<html>ok</html>"

    assert result.display() is None


def test_animation_result_to_html_uses_save_pipeline(monkeypatch, ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False, "print": False})
    calls = []

    def fake_save(path, writer=None, progress_callback=None, **kwargs):
        calls.append((writer.__class__.__name__, callable(progress_callback)))
        Path(path).write_text("<html>ok</html>", encoding="utf-8")
        if callable(progress_callback):
            progress_callback(0, 1)

    from pathlib import Path

    monkeypatch.setattr(result.animation, "save", fake_save)

    html = result.to_html()

    assert html == "<html>ok</html>"
    assert calls == [("HTMLWriter", True)]


def test_animation_result_save_uses_requested_path_and_format(monkeypatch, tmp_path, ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    calls = []

    def fake_save(path, writer=None, **kwargs):
        calls.append((path, writer, kwargs))

    monkeypatch.setattr(result.animation, "save", fake_save)
    target = tmp_path / "movie.gif"

    returned = result.save(target)

    assert calls
    assert str(target) in str(calls[0][0])
    assert calls[0][1] == "pillow"
    assert returned == target


def test_animation_result_save_filters_non_matplotlib_plot_options(
    monkeypatch,
    tmp_path,
    ecat_module,
    cv_factory,
):
    result = ecat_module.animate(cv_factory(), {"plot": False, "plot all": True, "legend": True})
    calls = []

    def fake_save(path, writer=None, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(result.animation, "save", fake_save)

    result.save(tmp_path / "movie.gif", options={"plot all": True, "legend": True, "dpi": 150})

    assert calls == [{"progress_callback": ANY, "fps": 20, "dpi": 150}]


def test_animation_result_save_preserves_explicit_savefig_kwargs_without_bbox_default(
    monkeypatch,
    tmp_path,
    ecat_module,
    cv_factory,
):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    calls = []

    def fake_save(path, writer=None, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(result.animation, "save", fake_save)

    result.save(
        tmp_path / "movie.gif",
        options={"dpi": 150, "savefig kwargs": {"facecolor": "white"}},
    )

    assert calls == [
        {
            "progress_callback": ANY,
            "fps": 20,
            "dpi": 150,
            "savefig_kwargs": {"facecolor": "white"},
        }
    ]


def test_animation_result_save_can_infer_mp4_writer(monkeypatch, tmp_path, ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    calls = []

    def fake_save(path, writer=None, **kwargs):
        calls.append(writer)

    monkeypatch.setattr(result.animation, "save", fake_save)

    result.save(tmp_path / "movie.mp4")

    assert calls == ["ffmpeg"]


def test_animation_result_save_can_write_html(monkeypatch, tmp_path, ecat_module, cv_factory):
    result = ecat_module.animate(cv_factory(), {"plot": False})
    called = []

    def fake_to_html(options=None):
        called.append(options)
        return "<html>saved</html>"

    monkeypatch.setattr(result, "to_html", fake_to_html)

    target = tmp_path / "movie.html"
    returned = result.save(target)

    assert returned == target
    assert target.read_text(encoding="utf-8") == "<html>saved</html>"
    assert called == [None]


def test_animation_notebook_progress_uses_count_over_total_frame_text(ecat_module):
    animation_mod = ecat_module.animation

    html_mid = animation_mod._NotebookAnimationProgressDisplay._html(
        2,
        5,
        "Rendering animation",
        elapsed=4.0,
        remaining=6.0,
    )
    assert "2 / 5 frames (40%)" in html_mid
    assert "remaining ~6.0s" in html_mid

    html_done = animation_mod._NotebookAnimationProgressDisplay._done_html(
        5,
        5,
        "Rendering animation",
        elapsed=10.0,
    )
    assert "5 / 5 frames" in html_done
    assert "remaining" not in html_done


def test_animation_notebook_progress_uses_indeterminate_when_total_unknown(ecat_module):
    animation_mod = ecat_module.animation

    html_indeterminate = animation_mod._NotebookAnimationProgressDisplay._html(
        3,
        None,
        None,
        "Rendering animation",
        elapsed=1.2,
        remaining=2.0,
    )
    assert "3 frames" in html_indeterminate
    assert "5 / 5 frames" not in html_indeterminate
    assert "ecat-indeterminate" in html_indeterminate

    html_done = animation_mod._NotebookAnimationProgressDisplay._done_html(
        3,
        None,
        None,
        "Rendering animation",
        elapsed=2.0,
    )
    assert "width:100%" in html_done


def test_animation_summary_reports_normalized_duration_and_frame_count(ecat_module, cv_factory):
    result = ecat_module.animate(
        cv_factory(),
        {
            "plot": False,
            "timing mode": "normalized",
            "normalized duration": 2.5,
            "end hold": 0.5,
            "fps": 20,
        },
    )

    assert result.summary["normalized duration"] == 2.5
    assert result.summary["estimated animation time"] == pytest.approx(3.0)
    assert result.summary["frame count"] == 61


def test_animation_summary_reports_physical_speedup(ecat_module, blank_echem_factory):
    obj = blank_echem_factory()
    obj.data = pd.DataFrame({"Time/sec": [0.0, 1.0, 2.0], "Current (A)": [0.0, 1.0, 0.5]})
    obj.num_x_cols = 1
    obj.compounds = []
    obj.concentrations = []

    result = ecat_module.animate(
        obj,
        {
            "plot": False,
            "timing mode": "physical",
            "speedup": 2,
            "end hold": 0,
            "fps": 10,
        },
    )

    assert result.summary["speedup"] == 2
    assert result.summary["estimated animation time"] == pytest.approx(1.0)
    assert result.summary["frame count"] == 11


def test_animation_setup_table_shows_auto_resolution(ecat_module, cv_factory):
    import ecat.animation as animation_module

    slow = cv_factory(name="50mVs_sample_CO2_MeCN_10mM_Fc_run01")
    fast = cv_factory(name="100mVs_sample_CO2_MeCN_10mM_Fc_run01")

    result = ecat_module.animate(
        [slow, fast],
        {"plot": False, "trace mode": "auto", "schedule": "auto", "timing mode": "auto"},
    )
    table = animation_module._setup_table(result.summary)

    values = dict(zip(table["Setting"], table["Value"]))

    assert values["Timing Mode"] == "auto -> physical"
    assert values["Trace Mode"] == "auto -> draw"
    assert values["Schedule"] == "auto -> simultaneous"
    assert values["Loop"] == "Yes"
    assert values["Include Quiet Time"] == "No"


def test_animate_with_plot_true_auto_displays_and_closes_static_figure(monkeypatch, ecat_module, cv_factory):
    import ecat.animation as animation_module

    calls = []

    def fake_display(self, options=None):
        calls.append(options)
        return self

    monkeypatch.setattr(animation_module.AnimationResult, "display", fake_display)
    plt.close("all")

    result = ecat_module.animate(cv_factory(), {"plot": True, "print": False})

    assert calls
    assert result.figure.number not in plt.get_fignums()


def _figure_content_bbox_in_figure_coords(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = []

    for ax in fig.axes:
        try:
            bbox = ax.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is not None and np.isfinite(bbox.bounds).all():
            bboxes.append(bbox)

    if fig._suptitle is not None:
        bbox = fig._suptitle.get_window_extent(renderer)
        if bbox is not None and np.isfinite(bbox.bounds).all():
            bboxes.append(bbox)

    assert bboxes, "Expected at least one figure-content bounding box."
    union = Bbox.union(bboxes)
    return union.transformed(fig.transFigure.inverted())


def test_animate_keeps_outside_legend_and_titles_inside_canvas(ecat_module, cv_factory):
    objects = [
        cv_factory(name="100mVs_sample_CO2_MeCN_0mM_PhOH_run01", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_100mM_PhOH_run02", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_560mM_PhOH_run03", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_1M_PhOH_run04", options={"electrode area": 0.071}),
        cv_factory(name="100mVs_sample_CO2_MeCN_2.8M_PhOH_run05", options={"electrode area": 0.071}),
    ]

    result = ecat_module.animate(
        objects,
        {
            "plot": False,
            "print": False,
            "title": "CO$_2$ Concentration-Series Animation",
            "subtitle": "PhOH titration sequence with normalized staggered playback",
            "plot labels": ["0 mM", "100 mM", "560 mM", "1 M", "2.8 M"],
            "legend": True,
            "legend outside": True,
            "y axis": "current density",
            "timing mode": "normalized",
            "stagger time": 0.35,
        },
    )

    bbox = _figure_content_bbox_in_figure_coords(result.figure)

    assert bbox.x0 >= -0.01
    assert bbox.y0 >= -0.01
    assert bbox.x1 <= 1.01
    assert bbox.y1 <= 1.01
