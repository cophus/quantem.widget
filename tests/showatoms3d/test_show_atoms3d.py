from __future__ import annotations

import numpy as np
import pytest

from quantem.widget.show_atoms3d import ShowAtoms3D


class _FakeModel:
    """Minimal stand-in for quantem.atoms.AtomicModel."""

    def __init__(self, n: int = 50) -> None:
        rng = np.random.default_rng(0)
        self.positions = rng.random((n, 3)) * 10.0
        self.units = "A"
        self.name = "fake"
        self.bond_length = 2.5
        self.channels = ["score", "structure"]
        self.categories = {"structure": ["fcc", "hcp"]}
        self._score = rng.random(n)
        self._structure = rng.integers(-1, 2, n).astype(float)

    def get_channel(self, name: str) -> np.ndarray:
        if name in ("x", "y", "z"):
            return self.positions[:, "xyz".index(name)]
        return {"score": self._score, "structure": self._structure}[name]


def test_construction_from_model_sends_positions_and_channels() -> None:
    model = _FakeModel(40)
    w = ShowAtoms3D(model)
    assert w.num_sites == 40
    assert w.channel == "structure"
    assert w.channel_names == ["x", "y", "z", "score", "structure"]
    assert np.frombuffer(w.positions, dtype=np.float32).reshape(40, 3).mean(0) == pytest.approx(0.0, abs=1e-5)
    data = np.frombuffer(w.channel_data, dtype=np.float32).reshape(5, 40)
    np.testing.assert_allclose(data[3], model._score.astype(np.float32))
    assert w.channel_categories == {"structure": ["fcc", "hcp"]}
    assert len(w.channel_palettes["structure"]) == 6
    assert w.bond_length == 2.5
    assert w.bounds[3] > w.bounds[0]


def test_construction_from_array() -> None:
    w = ShowAtoms3D(np.zeros((5, 3)) + np.arange(5)[:, None], title="t")
    assert w.channel_names == ["x", "y", "z"]
    assert w.channel == "x"
    assert w.title == "t"


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        ShowAtoms3D(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        ShowAtoms3D(_FakeModel(), channel="missing")
    with pytest.raises(ValueError):
        ShowAtoms3D(_FakeModel(), cmap="not-a-cmap")


def test_view_and_clip_helpers() -> None:
    w = ShowAtoms3D(_FakeModel())
    w.set_view("x")
    np.testing.assert_allclose(w.view_direction, [1, 0, 0])
    assert w.view_matrix.shape == (3, 3)
    w.clip(normal=(0, 0, 2), center=1.5, thickness=4.0)
    assert w.clip_enabled and w.clip_axis == "custom"
    assert w.clip_normal == [0.0, 0.0, 1.0]
    assert w.clip_thickness == 4.0
    w.set_range(0.2, 0.8)
    kwargs = w.slab_kwargs()
    assert kwargs["vmin"] == 0.2 and kwargs["channel"] == "structure"


def test_show_only_and_show_all() -> None:
    w = ShowAtoms3D(_FakeModel())
    w.show_only("structure", [1])
    assert w.hidden_categories == {"structure": [-1, 0]}
    w.show_all("structure")
    assert w.hidden_categories == {}
    with pytest.raises(ValueError):
        w.show_only("score", [0])


def test_selection_without_neighbors_is_graceful() -> None:
    w = ShowAtoms3D(_FakeModel())
    w.select(3)
    assert w.selected_info.startswith("site 3")
    assert "structure=" in w.selected_info
    assert "unavailable" in w.neighborhood_label
    w.select(-1)
    assert w.selected_info == "" and w.neighborhood == b""


def test_growth_slider() -> None:
    w = ShowAtoms3D(_FakeModel())
    w.growth("score")
    assert w.growth_channel == "score"
    assert w.growth_max == pytest.approx(float(np.max(w._model._score)), rel=1e-6)
    w.growth("score", value=0.3)
    assert w.growth_max == 0.3
    w.growth(None)
    assert w.growth_channel == ""
    with pytest.raises(ValueError):
        w.growth("missing")
