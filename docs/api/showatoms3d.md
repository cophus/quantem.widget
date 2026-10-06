# ShowAtoms3D

Interactive 3D viewer for atomic models with per-site measurement channels,
built for `quantem.atoms.AtomicModel`. Every site is drawn as a shaded sphere
colored by the selected channel. Drag to rotate, scroll to zoom, shift-drag to
pan, double-click to reset, and click a site to show its neighbors and matched
polyhedral template in the inspector.

```python
from quantem.widget import ShowAtoms3D

viewer = ShowAtoms3D(model, channel="structure")   # or model.show(...)
viewer
```

Scriptable state:

```python
viewer.set_view("z")                                   # look along a model axis
viewer.clip(axis="z", center=0.0, thickness=8.0)       # slab along a model axis
viewer.clip(normal=(0.3, 0.5, 0.8), thickness=6.0)     # slab with a custom normal
viewer.show_only("structure", [1])                     # only one class of a categorical channel
viewer.set_range(0.0, 0.8)                             # color range for a continuous channel
viewer.view_matrix                                     # rows: right, up, toward-viewer
model.plot("slab", **viewer.slab_kwargs())             # same view as a static figure
```

The custom clip normal can also be set by dragging the arrow handle drawn at
the clip plane, and legend entries of categorical channels toggle on click.
`viewer.selected` holds the index of the last picked site.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show_atoms3d.ShowAtoms3D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Channel dropdown | `channel` | Colors the sites by that per-site channel |
| Colormap dropdown | `cmap` | Colormap for continuous channels |
| Histogram handles / range inputs | `vmin`, `vmax`, `range_auto` | Color range; `auto` restores the 1st to 99th percentile |
| Hide outside | `hide_outside_range` | Hides sites whose value lies outside the range |
| Legend entry click | `hidden_categories` | Hides or shows one class of a categorical channel |
| Clip enable, axis, center, thickness | `clip_enabled`, `clip_axis`, `clip_center`, `clip_thickness` | Keeps only sites inside a slab |
| Arrow handle drag | `clip_normal` | Sets a custom clip normal (switches `clip_axis` to `custom`) |
| Growth channel and step slider | `growth_channel`, `growth_max` | Hides sites whose value in the channel exceeds the slider (`viewer.growth("shell")`) |
| Show plane, show bounds, axes | `show_clip_plane`, `show_bounds`, `show_axes` | Overlays for the clip planes, bounding box and axis triad |
| Marker, depth cue, edges, dark | `marker_size`, `depth_cue`, `show_edges`, `dark_background` | Sphere size, depth shading, outlines, canvas background |
| x / y / z / reset | `rotation`, `zoom`, `pan` | Preset views and reset |
| Site click | `selected`, `selected_info`, `neighborhood` | Picks a site; Python fills the inspector data |
