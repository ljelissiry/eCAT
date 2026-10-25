"""Plot labels combine solvent aliases without rewriting source metadata."""
import pytest


def _texts(ax):
    from matplotlib.text import Text
    return " ".join(text.get_text() for text in ax.figure.findobj(Text))


@pytest.mark.parametrize("labels,chosen,absent", [
    (["MeCN", "ACN", "ACN"], "ACN", "MeCN"),
    (["ACN", "MeCN", "MeCN"], "MeCN", "ACN"),
    (["ACN", "MeCN"], "ACN", "MeCN"),
    (["MeCN", "ACN"], "MeCN", "ACN"),
])
@pytest.mark.parametrize("gradient", ["scan rate", "concentration"])
def test_aliases_share_gradient_and_majority_label(ecat_module, cv_factory, labels, chosen, absent, gradient):
    objects = [cv_factory() for _ in labels]
    for index, (obj, label) in enumerate(zip(objects, labels)):
        obj.solvent = label
        if gradient == "scan rate":
            obj.scan_rate = 0.05 * (index + 1)
        else:
            obj.concentrations = [f"{index + 1} mM"]
    ax = ecat_module.multiplot(objects, {"print": False, "new plot": False, "gradient by": gradient, "min gradient entries": 2})
    assert len(ax.lines) == len(objects)
    assert len(ax.child_axes) == 1
    assert len(ax.child_axes[0].child_axes) == 1  # One shared colorbar.
    assert chosen in _texts(ax)
    assert absent not in _texts(ax)
    assert [obj.solvent for obj in objects] == labels


def test_different_solvents_remain_distinct_and_explicit_labels_survive(ecat_module, cv_factory):
    objects = [cv_factory() for _ in range(4)]
    for obj, label, rate in zip(objects, ["MeCN", "ACN", "ACN", "THF"], [.05, .1, .2, .1]):
        obj.solvent, obj.scan_rate = label, rate
    ax = ecat_module.multiplot(objects, {"print": False, "new plot": False, "legend mode": "discrete"})
    assert "ACN" in _texts(ax) and "THF" in _texts(ax)
    assert "MeCN" not in _texts(ax)
    labels = ["my MeCN trace", "second", "third", "THF control"]
    ax = ecat_module.multiplot(objects, {"print": False, "new plot": False, "legend mode": "discrete", "labels": labels})
    assert [text.get_text() for text in ax.get_legend().get_texts()] == labels


def test_filter_inspection_has_one_public_entry_point(ecat_module):
    import ecat
    from ecat import collection
    assert not hasattr(ecat, "get_available_filter_values")
    assert not hasattr(collection, "get_available_filter_values")
