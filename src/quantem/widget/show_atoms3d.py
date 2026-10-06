"""show_atoms3d: interactive 3D viewer for atomic models with per-site channels.

Displays every site of a :class:`quantem.atoms.AtomicModel` (or a plain
``(N, 3)`` coordinate array) as a shaded sphere, colored by any per-site
channel.  The browser renders and handles rotation, zoom, slab clipping and
picking; Python owns the coordinates, channels, categorical palettes and the
neighborhood data shown for a picked site.
"""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

import anywidget
import numpy as np
import traitlets
from numpy.typing import NDArray

__all__ = ["ShowAtoms3D"]

# Names known to the JS bundle (js/colormaps.ts COLORMAPS table).
_VALID_CMAPS = (
    "viridis",
    "turbo",
    "inferno",
    "magma",
    "plasma",
    "cividis",
    "RdBu_r",
    "RdBu",
    "seismic",
    "hot",
    "gray",
    "twilight",
    "hsv",
)

_CATEGORICAL_COLORS = np.array(
    [
        [0.20, 0.45, 0.95],
        [0.95, 0.60, 0.10],
        [0.20, 0.70, 0.30],
        [0.85, 0.20, 0.25],
        [0.60, 0.35, 0.80],
        [0.55, 0.35, 0.20],
        [0.90, 0.45, 0.75],
        [0.50, 0.50, 0.50],
        [0.70, 0.75, 0.10],
        [0.10, 0.75, 0.80],
    ]
)

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _view_matrix(normal: str | Sequence[float], up: str | Sequence[float] | None = None) -> NDArray:
    """Rows (right, up, toward-viewer) for a viewing direction ``normal``."""
    n = np.asarray(_AXES[normal] if isinstance(normal, str) else normal, dtype=float)
    n = n / np.linalg.norm(n)
    if up is None:
        up_v = np.array(_AXES["y"]) if abs(n[1]) < 0.9 else np.array(_AXES["x"])
        if isinstance(normal, str) and normal == "y":
            up_v = np.array(_AXES["x"])
    else:
        up_v = np.asarray(_AXES[up] if isinstance(up, str) else up, dtype=float)
    u = up_v - n * (up_v @ n)
    u = u / np.linalg.norm(u)
    return np.stack([np.cross(u, n), u, n], axis=0)


def _categorical_palette(name: str, n_cat: int) -> list[int]:
    if name == "grain" or n_cat > len(_CATEGORICAL_COLORS):
        rng = np.random.default_rng(0)
        colors = rng.uniform(0.15, 0.95, (max(n_cat, 1), 3))
    else:
        colors = _CATEGORICAL_COLORS[np.arange(max(n_cat, 1)) % len(_CATEGORICAL_COLORS)]
    return [int(v) for v in (colors * 255).round().astype(int).ravel()]


class ShowAtoms3D(anywidget.AnyWidget):
    """Interactive 3D viewer for an atomic model with per-site channels.

    Parameters
    ----------
    model : AtomicModel or array_like
        A :class:`quantem.atoms.AtomicModel`, or an ``(N, 3)`` coordinate
        array.  With a model, every channel is available in the channel
        dropdown; categorical channels (``structure``, ``grain``, ``boundary``,
        ``species``) get a legend whose entries toggle visibility on click.
    channel : str, optional
        Initial color channel.  Defaults to ``"structure"`` when present.
    cmap : str, default "viridis"
        Colormap for continuous channels (one of the frontend colormaps).
    canvas_size : int, default 640
        Canvas width and height in pixels.
    marker_size : float, default 0.7
        Sphere diameter as a fraction of the bond length.
    dark_background : bool, default True
        Dark canvas background.
    title : str, optional
        Title above the canvas; defaults to the model name.

    Notes
    -----
    Mouse: drag rotates, scroll zooms, shift-drag or right-drag pans,
    double-click resets, click picks a site.  With the site inspector enabled
    (``show_inspector``, off by default) the picked site is marked and its
    neighborhood and matched template are drawn in a small panel that can be
    rotated independently by dragging.  The clip controls cut a slab
    along the view direction, a model axis, or a custom normal; the custom
    normal is set by dragging the arrow handle drawn at the clip plane.
    The growth slider (:meth:`growth`) hides sites whose value in a chosen
    channel exceeds the slider, which replays a growth sequence stored per
    site.  The current view is available as :attr:`view_matrix` and
    :meth:`slab_kwargs` reproduces it with ``model.plot("slab", ...)``.

    Examples
    --------
    >>> viewer = ShowAtoms3D(model, channel="structure")
    >>> viewer
    >>> viewer.show_only("structure", [1])      # only hcp (twin) sites
    >>> viewer.clip("custom", normal=(0, 0, 1), thickness=6.0)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showatoms3d.js"

    # --- data (Python -> browser) ---------------------------------------- #
    positions = traitlets.Bytes(b"").tag(sync=True)
    num_sites = traitlets.Int(0).tag(sync=True)
    bounds = traitlets.List(traitlets.Float(), default_value=[0, 0, 0, 0, 0, 0]).tag(sync=True)
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_data = traitlets.Bytes(b"").tag(sync=True)
    channel_categories = traitlets.Dict().tag(sync=True)
    channel_palettes = traitlets.Dict().tag(sync=True)
    cmap_names = traitlets.List(traitlets.Unicode(), default_value=list(_VALID_CMAPS)).tag(sync=True)
    model_radius = traitlets.Float(1.0).tag(sync=True)
    bond_length = traitlets.Float(1.0).tag(sync=True)
    units = traitlets.Unicode("").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    # --- display state ---------------------------------------------------- #
    channel = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("viridis").tag(sync=True)
    vmin = traitlets.Float(0.0).tag(sync=True)
    vmax = traitlets.Float(1.0).tag(sync=True)
    range_auto = traitlets.Bool(True).tag(sync=True)
    hide_outside_range = traitlets.Bool(False).tag(sync=True)
    hidden_categories = traitlets.Dict().tag(sync=True)
    rotation = traitlets.List(traitlets.Float(), default_value=[1, 0, 0, 0, 1, 0, 0, 0, 1]).tag(sync=True)
    zoom = traitlets.Float(1.0).tag(sync=True)
    pan = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    clip_enabled = traitlets.Bool(False).tag(sync=True)
    clip_axis = traitlets.Enum(["view", "x", "y", "z", "custom"], default_value="view").tag(sync=True)
    clip_normal = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 1.0]).tag(sync=True)
    clip_center = traitlets.Float(0.0).tag(sync=True)
    clip_thickness = traitlets.Float(1.0).tag(sync=True)
    show_bounds = traitlets.Bool(True).tag(sync=True)
    show_clip_plane = traitlets.Bool(True).tag(sync=True)
    show_axes = traitlets.Bool(True).tag(sync=True)
    show_inspector = traitlets.Bool(False).tag(sync=True)
    growth_channel = traitlets.Unicode("").tag(sync=True)
    growth_max = traitlets.Float(0.0).tag(sync=True)
    marker_size = traitlets.Float(0.7).tag(sync=True)
    depth_cue = traitlets.Float(0.5).tag(sync=True)
    show_edges = traitlets.Bool(True).tag(sync=True)
    dark_background = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(640).tag(sync=True)

    # --- picking (browser -> Python -> browser) --------------------------- #
    selected = traitlets.Int(-1).tag(sync=True)
    selected_info = traitlets.Unicode("").tag(sync=True)
    neighborhood = traitlets.Bytes(b"").tag(sync=True)
    neighborhood_label = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        model: Any,
        channel: str | None = None,
        cmap: str = "viridis",
        canvas_size: int = 640,
        marker_size: float = 0.7,
        dark_background: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if cmap not in _VALID_CMAPS:
            raise ValueError(f"Unknown cmap {cmap!r}. Valid: {list(_VALID_CMAPS)}")
        self._model = model if hasattr(model, "positions") else None
        xyz = np.asarray(model.positions if self._model is not None else model, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected an AtomicModel or an (N, 3) array, got shape {xyz.shape}.")
        self._center = xyz.mean(0) if xyz.shape[0] else np.zeros(3)
        centered = (xyz - self._center).astype(np.float32)
        radius = float(np.linalg.norm(centered, axis=1).max()) if xyz.shape[0] else 1.0
        bond = self._bond_length(radius)
        with self.hold_sync():
            self.positions = np.ascontiguousarray(centered).tobytes()
            self.num_sites = int(xyz.shape[0])
            lo, hi = (centered.min(0), centered.max(0)) if xyz.shape[0] else (np.zeros(3), np.zeros(3))
            self.bounds = [float(v) for v in np.r_[lo, hi]]
            self.model_radius = radius
            self.bond_length = bond
            self.units = str(getattr(self._model, "units", ""))
            self.title = title if title is not None else str(getattr(self._model, "name", ""))
            self.cmap = cmap
            self.canvas_size = int(canvas_size)
            self.marker_size = float(marker_size)
            self.dark_background = bool(dark_background)
            self.clip_thickness = float(3.0 * bond)
            self.update_channels()
            if channel is None:
                channel = "structure" if "structure" in self.channel_names else self.channel_names[0]
            if channel not in self.channel_names:
                raise ValueError(f"Unknown channel {channel!r}. Available: {self.channel_names}")
            self.channel = channel
        self.observe(self._on_selected, names="selected")

    # ------------------------------------------------------------------ #
    def _bond_length(self, radius: float) -> float:
        if self._model is not None:
            try:
                return float(self._model.bond_length)
            except Exception:
                pass
        return float(radius / 20.0)

    def update_channels(self) -> None:
        """Re-send every channel of the model (call after computing new channels)."""
        if self._model is None:
            names = ["x", "y", "z"]
            data = np.frombuffer(self.positions, dtype=np.float32).reshape(-1, 3).T.copy()
            cats: dict[str, list[str]] = {}
            palettes: dict[str, list[int]] = {}
        else:
            m = self._model
            names = ["x", "y", "z"] + list(m.channels)
            data = np.stack([m.get_channel(n) for n in names], axis=0).astype(np.float32)
            cats = {}
            palettes = {}
            for name, labels in m.categories.items():
                values = m.get_channel(name)
                n_cat = max(len(labels), int(np.nanmax(values)) + 1 if values.size else 1)
                cats[name] = [str(c) for c in labels]
                palettes[name] = _categorical_palette(name, n_cat)
        with self.hold_sync():
            self.channel_names = names
            self.channel_data = np.ascontiguousarray(data).tobytes()
            self.channel_categories = cats
            self.channel_palettes = palettes

    # ------------------------------------------------------------------ #
    # view helpers
    # ------------------------------------------------------------------ #
    @property
    def view_matrix(self) -> NDArray:
        """``(3, 3)`` view matrix with rows (right, up, toward-viewer)."""
        return np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def view_direction(self) -> NDArray:
        """``(3,)`` unit vector from the model toward the viewer."""
        return self.view_matrix[2]

    def set_view(self, normal: str | Sequence[float], up: str | Sequence[float] | None = None) -> None:
        """Look along ``normal`` (``"x"``, ``"y"``, ``"z"`` or a 3-vector)."""
        self.rotation = [float(v) for v in _view_matrix(normal, up).ravel()]

    def set_range(self, vmin: float, vmax: float) -> None:
        """Set the color range and disable auto-ranging."""
        with self.hold_sync():
            self.range_auto = False
            self.vmin, self.vmax = float(vmin), float(vmax)

    def clip(
        self,
        axis: str = "view",
        center: float = 0.0,
        thickness: float | None = None,
        normal: Sequence[float] | None = None,
    ) -> None:
        """Enable slab clipping.

        Parameters
        ----------
        axis : {"view", "x", "y", "z", "custom"}
            Slab normal.  ``"custom"`` uses ``normal``.
        center : float
            Slab center along the normal, relative to the model center.
        thickness : float, optional
            Slab thickness in model units.
        normal : sequence of float, optional
            Custom normal (implies ``axis="custom"``).
        """
        with self.hold_sync():
            if normal is not None:
                n = np.asarray(normal, dtype=float)
                self.clip_normal = [float(v) for v in n / np.linalg.norm(n)]
                axis = "custom"
            self.clip_axis = axis
            self.clip_center = float(center)
            if thickness is not None:
                self.clip_thickness = float(thickness)
            self.clip_enabled = True

    def growth(self, channel: str | None = "growth_step", value: float | None = None) -> None:
        """Enable the growth slider on a per-site channel.

        Sites whose ``channel`` value exceeds the slider value are hidden, so
        stepping the slider replays the growth sequence encoded in the
        channel (for example a shell index).  ``channel=None`` disables it.

        Parameters
        ----------
        channel : str, optional
            Channel with the growth step of each site.
        value : float, optional
            Initial slider value; default: the channel maximum (everything shown).
        """
        if channel is None:
            self.growth_channel = ""
            return
        if channel not in self.channel_names:
            raise ValueError(f"Unknown channel {channel!r}. Available: {self.channel_names}")
        i = self.channel_names.index(channel)
        data = np.frombuffer(self.channel_data, dtype=np.float32).reshape(len(self.channel_names), -1)
        with self.hold_sync():
            self.growth_channel = channel
            self.growth_max = float(np.nanmax(data[i])) if value is None else float(value)

    def show_only(self, channel: str, codes: Sequence[int]) -> None:
        """Show only the listed category codes of a categorical channel."""
        labels = self.channel_categories.get(channel)
        if labels is None:
            raise ValueError(f"{channel!r} is not a categorical channel.")
        keep = {int(c) for c in codes}
        hidden = dict(self.hidden_categories)
        hidden[channel] = [c for c in range(-1, len(labels)) if c not in keep]
        with self.hold_sync():
            self.channel = channel
            self.hidden_categories = hidden

    def show_all(self, channel: str | None = None) -> None:
        """Clear hidden categories for one channel (or all channels)."""
        if channel is None:
            self.hidden_categories = {}
        else:
            hidden = dict(self.hidden_categories)
            hidden.pop(channel, None)
            self.hidden_categories = hidden

    def slab_kwargs(self) -> dict[str, Any]:
        """Keyword arguments reproducing the current view with ``model.plot("slab", ...)``."""
        vm = self.view_matrix
        out: dict[str, Any] = {"normal": vm[2], "up": vm[1], "channel": self.channel}
        if self.clip_enabled and self.clip_axis == "view":
            out.update(offset=self.clip_center, thickness=self.clip_thickness)
        if not self.range_auto:
            out.update(vmin=self.vmin, vmax=self.vmax)
        return out

    # ------------------------------------------------------------------ #
    # picking
    # ------------------------------------------------------------------ #
    def _on_selected(self, change: Any) -> None:
        index = int(change["new"])
        if index < 0 or self._model is None or index >= self.num_sites:
            with self.hold_sync():
                self.selected_info = ""
                self.neighborhood = b""
                self.neighborhood_label = ""
            return
        m = self._model
        info = [f"site {index}"]
        for name in m.channels:
            if name in m.categories:
                code = int(m.get_channel(name)[index])
                labels = m.categories[name]
                info.append(f"{name}={labels[code] if 0 <= code < len(labels) else code}")
            elif not name.startswith("matched_"):
                info.append(f"{name}={m.get_channel(name)[index]:.3g}")
        payload = b""
        label = ""
        try:
            dxyz, dist = m.neighbor_vectors(normalize=False)
            scale = float(np.mean(m.sampling))
            template_name = None
            if m.matches:
                if "structure" in m.channels:
                    code = int(m.get_channel("structure")[index])
                    if code >= 0:
                        template_name = m.structure_names[code]
                if template_name is None:
                    template_name = max(m.matches, key=lambda n: m.matches[n]["score"][index])
                template = m.templates[template_name]
                keep = np.isfinite(dist[index]) & (dist[index] <= 1.3 * template.max_radius * m.nn_distance)
                nb = dxyz[index][keep] * scale
                rot = m.matches[template_name]["rotation"][index]
                tv = (template.vectors @ rot.T) * m.nn_distance * scale
                score = float(m.matches[template_name]["score"][index])
                label = f"{template_name}  score {score:.2f}  matched {int(m.matches[template_name]['num_matched'][index])}/{template.num_neighbors}"
            else:
                keep = np.isfinite(dist[index]) & (dist[index] <= 1.3 * m.first_shell_cutoff)
                nb = dxyz[index][keep] * scale
                tv = np.zeros((0, 3))
                label = f"{int(keep.sum())} neighbors"
            header = np.array([nb.shape[0], tv.shape[0], 0.0], dtype=np.float32)
            payload = np.concatenate([header, nb.astype(np.float32).ravel(), tv.astype(np.float32).ravel()]).tobytes()
        except Exception as exc:  # neighborhood is optional
            label = f"neighborhood unavailable ({exc})"
        with self.hold_sync():
            self.selected_info = "  ".join(info)
            self.neighborhood = payload
            self.neighborhood_label = label

    def select(self, index: int) -> None:
        """Select a site by index (as clicking it in the browser would)."""
        self.selected = int(index)

    def __repr__(self) -> str:
        return (
            f"ShowAtoms3D({self.num_sites} sites, channel={self.channel!r}, "
            f"clip={self.clip_axis if self.clip_enabled else 'off'})"
        )
