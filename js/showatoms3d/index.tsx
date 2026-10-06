/**
 * ShowAtoms3D - interactive 3D viewer for atomic models with per-site channels.
 *
 * One canvas draws every site as a shaded sprite, painter-sorted by depth.
 * Drag rotates, scroll zooms, shift/right-drag pans, double-click resets, and a
 * click picks a site.  A channel dropdown colors the sites; the histogram's two
 * handles set the color range and "hide outside range" removes sites outside it.
 * Legend entries of categorical channels toggle on click.  The clip controls
 * cut a slab along the view direction, a model axis, or a custom normal whose
 * arrow handle can be dragged on the canvas.  A second small canvas shows the
 * neighborhood and matched template of the picked site (data from Python).
 */

import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import { useTheme } from "../theme";
import { extractFloat32 } from "../format";
import { COLORMAPS } from "../colormaps";
import {
  IDENTITY, Mat3, Vec3, clipNormal, dragNormal, mat3apply, mat3mul,
  orthonormalize, planeBoxPolygon, projectPositions, rotX, rotY, viewMatrix,
} from "./geometry";

const HIST_BINS = 64;
const HANDLE_HIT_PX = 14;
const NEIGHBOR_CANVAS = 220;

const sliderSx = {
  py: 0,
  "& .MuiSlider-thumb": { width: 11, height: 11 },
  "& .MuiSlider-rail": { height: 2 },
  "& .MuiSlider-track": { height: 2 },
};
const compactButton = { fontSize: 10, py: 0.25, px: 1, minWidth: 0, textTransform: "none" as const };

type Categories = Record<string, string[]>;
type Palettes = Record<string, number[]>;
type Hidden = Record<string, number[]>;
type DragMode = "none" | "rotate" | "pan" | "handle";

function rgbString(r: number, g: number, b: number, a = 1): string {
  return `rgba(${r | 0},${g | 0},${b | 0},${a})`;
}

/** Sprite cache: shaded discs keyed by color and depth shade. */
class SpriteCache {
  private cache = new Map<number, HTMLCanvasElement>();
  private radius = -1;
  private edges = true;
  get(r: number, g: number, b: number, shade: number, radiusPx: number, edges: boolean): HTMLCanvasElement {
    if (radiusPx !== this.radius || edges !== this.edges) { this.cache.clear(); this.radius = radiusPx; this.edges = edges; }
    const key = ((r << 16) | (g << 8) | b) * 32 + shade;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const R = Math.max(1, Math.ceil(radiusPx));
    const size = 2 * R + 2;
    const s = document.createElement("canvas");
    s.width = size; s.height = size;
    const c = s.getContext("2d")!;
    const f = 1 - 0.85 * shade / 31;
    const cx = R + 1, cy = R + 1;
    const grad = c.createRadialGradient(cx - 0.35 * R, cy - 0.35 * R, 0.1 * R, cx, cy, R);
    grad.addColorStop(0, rgbString(Math.min(255, r * f + 90 * f), Math.min(255, g * f + 90 * f), Math.min(255, b * f + 90 * f)));
    grad.addColorStop(1, rgbString(r * f * 0.75, g * f * 0.75, b * f * 0.75));
    c.fillStyle = grad;
    c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.fill();
    if (edges && R > 2) { c.strokeStyle = "rgba(0,0,0,0.6)"; c.lineWidth = Math.max(0.5, R * 0.12); c.stroke(); }
    this.cache.set(key, s);
    return s;
  }
}

function ShowAtoms3D() {
  const model = useModel();
  const { colors: theme } = useTheme();

  // ---- data traits ---------------------------------------------------------
  const [numSites] = useModelState<number>("num_sites");
  const [positionsView] = useModelState<DataView>("positions");
  const [bounds] = useModelState<number[]>("bounds");
  const [channelNames] = useModelState<string[]>("channel_names");
  const [channelDataView] = useModelState<DataView>("channel_data");
  const [categories] = useModelState<Categories>("channel_categories");
  const [palettes] = useModelState<Palettes>("channel_palettes");
  const [cmapNames] = useModelState<string[]>("cmap_names");
  const [modelRadius] = useModelState<number>("model_radius");
  const [bondLength] = useModelState<number>("bond_length");
  const [units] = useModelState<string>("units");
  const [title] = useModelState<string>("title");

  // ---- display traits ------------------------------------------------------
  const [channel, setChannel] = useModelState<string>("channel");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [vmin, setVmin] = useModelState<number>("vmin");
  const [vmax, setVmax] = useModelState<number>("vmax");
  const [rangeAuto, setRangeAuto] = useModelState<boolean>("range_auto");
  const [hideOutside, setHideOutside] = useModelState<boolean>("hide_outside_range");
  const [hidden, setHidden] = useModelState<Hidden>("hidden_categories");
  const [rotation, setRotation] = useModelState<number[]>("rotation");
  const [zoom, setZoom] = useModelState<number>("zoom");
  const [pan, setPan] = useModelState<number[]>("pan");
  const [clipEnabled, setClipEnabled] = useModelState<boolean>("clip_enabled");
  const [clipAxis, setClipAxis] = useModelState<string>("clip_axis");
  const [clipNormalT, setClipNormalT] = useModelState<number[]>("clip_normal");
  const [clipCenter, setClipCenter] = useModelState<number>("clip_center");
  const [clipThickness, setClipThickness] = useModelState<number>("clip_thickness");
  const [showBounds, setShowBounds] = useModelState<boolean>("show_bounds");
  const [showClipPlane, setShowClipPlane] = useModelState<boolean>("show_clip_plane");
  const [showAxes, setShowAxes] = useModelState<boolean>("show_axes");
  const [showInspector, setShowInspector] = useModelState<boolean>("show_inspector");
  const [markerSize, setMarkerSize] = useModelState<number>("marker_size");
  const [depthCue, setDepthCue] = useModelState<number>("depth_cue");
  const [showEdges, setShowEdges] = useModelState<boolean>("show_edges");
  const [darkBackground, setDarkBackground] = useModelState<boolean>("dark_background");
  const [canvasSize] = useModelState<number>("canvas_size");
  const [growthChannel, setGrowthChannel] = useModelState<string>("growth_channel");
  const [growthMax, setGrowthMax] = useModelState<number>("growth_max");
  const [selected, setSelected] = useModelState<number>("selected");
  const [selectedInfo] = useModelState<string>("selected_info");
  const [neighborhoodView] = useModelState<DataView>("neighborhood");
  const [neighborhoodLabel] = useModelState<string>("neighborhood_label");

  // ---- decoded data (memoized) --------------------------------------------
  const positions = React.useMemo(() => extractFloat32(positionsView) ?? new Float32Array(0), [positionsView]);
  const channelData = React.useMemo(() => extractFloat32(channelDataView) ?? new Float32Array(0), [channelDataView]);
  const neighborhood = React.useMemo(() => extractFloat32(neighborhoodView) ?? new Float32Array(0), [neighborhoodView]);
  const channelIndex = Math.max(0, channelNames.indexOf(channel));
  const values = React.useMemo(
    () => channelData.subarray(channelIndex * numSites, (channelIndex + 1) * numSites),
    [channelData, channelIndex, numSites],
  );
  const growthIndex = channelNames.indexOf(growthChannel);
  const growthValues = React.useMemo(
    () => (growthIndex >= 0 ? channelData.subarray(growthIndex * numSites, (growthIndex + 1) * numSites) : null),
    [channelData, growthIndex, numSites],
  );
  const growthRange = React.useMemo(() => {
    if (!growthValues) return { lo: 0, hi: 1, step: 1 };
    let lo = Infinity, hi = -Infinity, integer = true;
    for (let i = 0; i < growthValues.length; i++) {
      const v = growthValues[i]; if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v; if (v > hi) hi = v; if (integer && Math.abs(v - Math.round(v)) > 1e-6) integer = false;
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    return { lo, hi: hi > lo ? hi : lo + 1, step: integer ? 1 : (hi - lo) / 200 };
  }, [growthValues]);
  const isCategorical = categories[channel] !== undefined;
  const palette = palettes[channel];
  const hiddenCodes = React.useMemo(() => new Set(hidden[channel] ?? []), [hidden, channel]);

  // ---- working buffers -----------------------------------------------------
  const proj = React.useRef(new Float32Array(0));
  const order = React.useRef(new Uint32Array(0));
  const visible = React.useRef(new Uint8Array(0));
  const sprites = React.useRef(new SpriteCache());
  if (proj.current.length !== numSites * 3) {
    proj.current = new Float32Array(numSites * 3);
    order.current = new Uint32Array(numSites);
    visible.current = new Uint8Array(numSites);
  }

  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const histRef = React.useRef<HTMLCanvasElement>(null);
  const nbRef = React.useRef<HTMLCanvasElement>(null);
  const dragRef = React.useRef<{ mode: DragMode; x: number; y: number; moved: boolean }>({ mode: "none", x: 0, y: 0, moved: false });
  const histDragRef = React.useRef<0 | 1 | 2>(0);
  // the inspector keeps its own view, seeded from the main view whenever a new site is picked
  const [nbView, setNbView] = React.useState<Mat3>(rotation as Mat3);
  const nbDragRef = React.useRef<{ x: number; y: number } | null>(null);
  React.useEffect(() => { setNbView(rotation as Mat3); }, [neighborhoodView]); // eslint-disable-line react-hooks/exhaustive-deps
  const handleScreenRef = React.useRef<{ x: number; y: number } | null>(null);

  // ---- histogram + auto range ---------------------------------------------
  const hist = React.useMemo(() => {
    let lo: number, hi: number;
    if (isCategorical) { lo = -1.5; hi = (categories[channel]?.length ?? 1) - 0.5; }
    else {
      lo = Infinity; hi = -Infinity;
      for (let i = 0; i < values.length; i++) { const x = values[i]; if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; } }
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      if (hi <= lo) hi = lo + 1e-9;
    }
    const counts = new Float64Array(HIST_BINS);
    for (let i = 0; i < values.length; i++) {
      const x = values[i]; if (!Number.isFinite(x)) continue;
      let b = Math.floor((x - lo) / (hi - lo) * HIST_BINS); if (b >= HIST_BINS) b = HIST_BINS - 1; if (b < 0) b = 0;
      counts[b]++;
    }
    return { lo, hi, counts };
  }, [values, isCategorical, categories, channel]);

  const autoRange = React.useCallback(() => {
    if (isCategorical) { setVmin(-0.5); setVmax((categories[channel]?.length ?? 1) - 0.5); return; }
    const arr = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
    if (!arr.length) return;
    const lo = arr[Math.floor(0.01 * (arr.length - 1))], hi = arr[Math.floor(0.99 * (arr.length - 1))];
    setVmin(lo); setVmax(hi > lo ? hi : lo + 1e-9);
  }, [values, isCategorical, categories, channel, setVmin, setVmax]);

  React.useEffect(() => { if (rangeAuto) autoRange(); }, [channel, rangeAuto, values]); // eslint-disable-line react-hooks/exhaustive-deps

  const colorOf = React.useCallback((val: number): [number, number, number] => {
    if (isCategorical) {
      const code = Math.round(val);
      if (!Number.isFinite(val) || code < 0 || !palette) return [140, 140, 140];
      const j = 3 * (code % (palette.length / 3));
      return [palette[j], palette[j + 1], palette[j + 2]];
    }
    if (!Number.isFinite(val)) return [140, 140, 140];
    const lut = COLORMAPS[cmap] ?? COLORMAPS["viridis"];
    let t = (val - vmin) / Math.max(vmax - vmin, 1e-12);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const o = Math.round(t * 255) * 3;
    return [lut[o], lut[o + 1], lut[o + 2]];
  }, [isCategorical, palette, cmap, vmin, vmax]);

  const isHidden = React.useCallback((val: number): boolean => {
    if (isCategorical) {
      const code = Number.isFinite(val) ? Math.round(val) : -1;
      if (hiddenCodes.has(code)) return true;
    }
    if (hideOutside && !(val >= vmin && val <= vmax)) return true;
    return false;
  }, [isCategorical, hiddenCodes, hideOutside, vmin, vmax]);

  // ---- derived view quantities --------------------------------------------
  const view = rotation as Mat3;
  const scale = zoom * (0.5 * canvasSize * 0.92) / Math.max(modelRadius, 1e-9);
  const cx = canvasSize / 2 + pan[0], cy = canvasSize / 2 + pan[1];
  const clipN = clipNormal(clipAxis, view, clipNormalT as Vec3);
  const toScreen = React.useCallback((p: Vec3): [number, number, number] => {
    const q = mat3apply(view, p);
    return [cx + q[0] * scale, cy - q[1] * scale, q[2]];
  }, [view, cx, cy, scale]);

  // ---- main draw -----------------------------------------------------------
  React.useLayoutEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    canvas.width = canvasSize; canvas.height = canvasSize;
    ctx.fillStyle = darkBackground ? "#111" : "#fff";
    ctx.fillRect(0, 0, canvasSize, canvasSize);
    const lo: Vec3 = [bounds[0], bounds[1], bounds[2]], hi: Vec3 = [bounds[3], bounds[4], bounds[5]];

    // back faces of the bounding box and clip planes are drawn first
    const lineColor = darkBackground ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.35)";
    const drawPolyline = (pts: Vec3[], color: string, close: boolean, fill?: string) => {
      if (!pts.length) return;
      ctx.beginPath();
      pts.forEach((p, i) => { const s = toScreen(p); if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]); });
      if (close) ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
    };
    if (showBounds) {
      const c = (i: number): Vec3 => [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]];
      const edges: [number, number][] = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
      ctx.strokeStyle = lineColor; ctx.lineWidth = 1;
      for (const [a, b] of edges) { const s0 = toScreen(c(a)), s1 = toScreen(c(b)); ctx.beginPath(); ctx.moveTo(s0[0], s0[1]); ctx.lineTo(s1[0], s1[1]); ctx.stroke(); }
    }
    const planeFill = darkBackground ? "rgba(255,180,80,0.12)" : "rgba(220,120,0,0.12)";
    const planeLine = darkBackground ? "rgba(255,180,80,0.7)" : "rgba(220,120,0,0.8)";
    if (clipEnabled && showClipPlane && clipAxis !== "view") {
      for (const off of [-clipThickness / 2, clipThickness / 2]) {
        drawPolyline(planeBoxPolygon(clipN, clipCenter + off, lo, hi), planeLine, true, planeFill);
      }
    }

    // sites
    const clip = clipEnabled ? { normal: clipN, center: clipCenter, thickness: clipThickness } : null;
    const { dmin, dmax } = projectPositions(positions, { view, scale, cx, cy, clip }, proj.current, visible.current);
    const vis = visible.current, pr = proj.current, ord = order.current;
    let count = 0;
    for (let i = 0; i < numSites; i++) {
      if (vis[i] && isHidden(values[i])) vis[i] = 0;
      if (vis[i] && growthValues && growthValues[i] > growthMax) vis[i] = 0;
      if (vis[i]) ord[count++] = i;
    }
    const sub = ord.subarray(0, count);
    sub.sort((a, b) => pr[3 * a + 2] - pr[3 * b + 2]);
    const radiusPx = Math.max(0.6, markerSize * bondLength * 0.5 * scale);
    const dr = Math.max(dmax - dmin, 1e-9);
    for (let k = 0; k < count; k++) {
      const i = sub[k];
      const [r, g, b] = colorOf(values[i]);
      const t = (pr[3 * i + 2] - dmin) / dr;
      const shade = Math.round(depthCue * (1 - t) * 31);
      const s = sprites.current.get(r, g, b, shade, radiusPx, showEdges);
      ctx.drawImage(s, pr[3 * i] - s.width / 2, pr[3 * i + 1] - s.height / 2);
    }
    if (showInspector && selected >= 0 && selected < numSites && vis[selected]) {
      ctx.strokeStyle = darkBackground ? "#fff" : "#000"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(pr[3 * selected], pr[3 * selected + 1], radiusPx + 3, 0, 2 * Math.PI); ctx.stroke();
    }

    // clip-plane normal handle (arrow from the plane center along the normal)
    handleScreenRef.current = null;
    if (clipEnabled && showClipPlane && clipAxis !== "view") {
      const base: Vec3 = [clipN[0] * clipCenter, clipN[1] * clipCenter, clipN[2] * clipCenter];
      const len = 0.55 * modelRadius;
      const tip: Vec3 = [base[0] + clipN[0] * len, base[1] + clipN[1] * len, base[2] + clipN[2] * len];
      const s0 = toScreen(base), s1 = toScreen(tip);
      ctx.strokeStyle = planeLine; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(s0[0], s0[1]); ctx.lineTo(s1[0], s1[1]); ctx.stroke();
      ctx.fillStyle = planeLine;
      ctx.beginPath(); ctx.arc(s1[0], s1[1], 7, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = darkBackground ? "#fff" : "#000"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(s0[0], s0[1], 3, 0, 2 * Math.PI); ctx.fill();
      handleScreenRef.current = { x: s1[0], y: s1[1] };
    }

    // axes triad (bottom-left)
    if (showAxes) {
      const ox = 34, oy = canvasSize - 34, L = 24;
      const names = ["x", "y", "z"], cols = ["#f55", "#5d5", "#59f"];
      for (let a = 0; a < 3; a++) {
        const v: Vec3 = [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
        const q = mat3apply(view, v);
        ctx.strokeStyle = cols[a]; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + q[0] * L, oy - q[1] * L); ctx.stroke();
        ctx.fillStyle = cols[a]; ctx.font = "11px sans-serif";
        ctx.fillText(names[a], ox + q[0] * L * 1.35 - 3, oy - q[1] * L * 1.35 + 4);
      }
    }
    ctx.fillStyle = darkBackground ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${count} / ${numSites} sites`, 8, canvasSize - 8);
  });

  // ---- histogram draw ------------------------------------------------------
  React.useLayoutEffect(() => {
    const c = histRef.current; if (!c) return;
    const ctx = c.getContext("2d"); if (!ctx) return;
    const W = c.width, H = c.height, pad = 4, hh = H - 22;
    ctx.fillStyle = theme.bgAlt; ctx.fillRect(0, 0, W, H);
    const cmax = Math.max(1, ...hist.counts);
    for (let b = 0; b < HIST_BINS; b++) {
      const x0 = pad + (W - 2 * pad) * b / HIST_BINS, w = (W - 2 * pad) / HIST_BINS;
      const val = hist.lo + (b + 0.5) / HIST_BINS * (hist.hi - hist.lo);
      const [r, g, bb] = colorOf(val);
      const h = Math.log1p(hist.counts[b]) / Math.log1p(cmax) * hh;
      ctx.globalAlpha = (val < vmin || val > vmax || isHidden(val)) ? 0.3 : 1.0;
      ctx.fillStyle = rgbString(r, g, bb); ctx.fillRect(x0, pad + hh - h, Math.max(w - 1, 1), h);
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < W - 2 * pad; i++) {
      const val = hist.lo + i / (W - 2 * pad) * (hist.hi - hist.lo);
      const [r, g, bb] = colorOf(val);
      ctx.fillStyle = rgbString(r, g, bb); ctx.fillRect(pad + i, pad + hh + 4, 1, 8);
    }
    ctx.strokeStyle = theme.text; ctx.lineWidth = 1.5;
    for (const v of [vmin, vmax]) {
      const x = pad + (W - 2 * pad) * (v - hist.lo) / (hist.hi - hist.lo);
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - 2); ctx.stroke();
    }
  }, [hist, vmin, vmax, colorOf, isHidden, theme]);

  // ---- neighborhood draw ---------------------------------------------------
  React.useLayoutEffect(() => {
    const c = nbRef.current; if (!c) return;
    const ctx = c.getContext("2d"); if (!ctx) return;
    const S = c.width;
    ctx.fillStyle = darkBackground ? "#111" : "#fff"; ctx.fillRect(0, 0, S, S);
    if (neighborhood.length < 3) {
      ctx.fillStyle = theme.textMuted; ctx.font = "11px sans-serif";
      ctx.fillText("click a site", 8, S / 2);
      return;
    }
    const nNb = neighborhood[0] | 0, nT = neighborhood[1] | 0;
    const nb = neighborhood.subarray(3, 3 + 3 * nNb), tv = neighborhood.subarray(3 + 3 * nNb, 3 + 3 * nNb + 3 * nT);
    let rmax = 1e-9;
    for (let i = 0; i < nNb; i++) rmax = Math.max(rmax, Math.hypot(nb[3 * i], nb[3 * i + 1], nb[3 * i + 2]));
    const sc = (S / 2 - 14) / rmax;
    const pts: { x: number; y: number; d: number; kind: number }[] = [];
    for (let i = 0; i < nNb; i++) { const q = mat3apply(nbView, [nb[3 * i], nb[3 * i + 1], nb[3 * i + 2]]); pts.push({ x: S / 2 + q[0] * sc, y: S / 2 - q[1] * sc, d: q[2], kind: 0 }); }
    for (let i = 0; i < nT; i++) { const q = mat3apply(nbView, [tv[3 * i], tv[3 * i + 1], tv[3 * i + 2]]); pts.push({ x: S / 2 + q[0] * sc, y: S / 2 - q[1] * sc, d: q[2], kind: 1 }); }
    // neighbors back-to-front, then the template markers on top
    pts.sort((a, b) => (a.kind - b.kind) || (a.d - b.d));
    const rPx = Math.max(3, 0.18 * bondLength * sc);
    for (const p of pts) {
      if (p.kind === 0) {
        const f = 0.6 + 0.4 * (p.d / rmax + 1) / 2;
        ctx.fillStyle = rgbString(160 * f, 160 * f, 160 * f); ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p.x, p.y, rPx, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      } else {
        ctx.strokeStyle = "#f33"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p.x - rPx, p.y); ctx.lineTo(p.x + rPx, p.y); ctx.moveTo(p.x, p.y - rPx); ctx.lineTo(p.x, p.y + rPx); ctx.stroke();
      }
    }
    ctx.strokeStyle = "#39f"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(S / 2 - 5, S / 2); ctx.lineTo(S / 2 + 5, S / 2); ctx.moveTo(S / 2, S / 2 - 5); ctx.lineTo(S / 2, S / 2 + 5); ctx.stroke();
  }, [neighborhood, nbView, bondLength, darkBackground, theme, showInspector]);

  const onNbDown = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    nbDragRef.current = { x: ev.clientX, y: ev.clientY };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onNbMove = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const d = nbDragRef.current; if (!d) return;
    const dx = ev.clientX - d.x, dy = ev.clientY - d.y;
    d.x = ev.clientX; d.y = ev.clientY;
    const k = 0.01;
    setNbView(orthonormalize(mat3mul(mat3mul(rotX(dy * k), rotY(dx * k)), nbView)));
  };
  const onNbUp = (ev: React.PointerEvent<HTMLCanvasElement>) => { nbDragRef.current = null; ev.currentTarget.releasePointerCapture(ev.pointerId); };

  // ---- interaction ---------------------------------------------------------
  const localPos = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) * canvasSize / rect.width, y: (ev.clientY - rect.top) * canvasSize / rect.height };
  };
  const onPointerDown = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const p = localPos(ev);
    const h = handleScreenRef.current;
    let mode: DragMode = (ev.button === 2 || ev.shiftKey) ? "pan" : "rotate";
    if (h && Math.hypot(p.x - h.x, p.y - h.y) < HANDLE_HIT_PX && ev.button === 0 && !ev.shiftKey) mode = "handle";
    dragRef.current = { mode, x: ev.clientX, y: ev.clientY, moved: false };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onPointerMove = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current; if (d.mode === "none") return;
    const dx = ev.clientX - d.x, dy = ev.clientY - d.y;
    d.x = ev.clientX; d.y = ev.clientY;
    if (Math.abs(dx) + Math.abs(dy) > 0) d.moved = true;
    if (d.mode === "pan") setPan([pan[0] + dx, pan[1] + dy]);
    else if (d.mode === "handle") {
      const n = dragNormal(clipN, view, dx, dy, 0.6 / Math.max(0.55 * modelRadius * scale, 1));
      if (clipAxis !== "custom") setClipAxis("custom");
      setClipNormalT(n);
    } else {
      // screen-space trackball: horizontal drag spins about the screen up axis,
      // vertical drag about the screen right axis, following the mouse.
      const k = 0.008;
      const rs = mat3mul(rotX(dy * k), rotY(dx * k));
      setRotation(orthonormalize(mat3mul(rs, view)));
    }
  };
  const onPointerUp = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    ev.currentTarget.releasePointerCapture(ev.pointerId);
    if (d.mode !== "none" && !d.moved) pick(localPos(ev));
    dragRef.current = { mode: "none", x: 0, y: 0, moved: false };
    model.save_changes();
  };
  const pick = (p: { x: number; y: number }) => {
    const pr = proj.current, vis = visible.current;
    let best = -1, bd = Infinity;
    for (let i = 0; i < numSites; i++) {
      if (!vis[i]) continue;
      const dx = pr[3 * i] - p.x, dy = pr[3 * i + 1] - p.y;
      const d2 = dx * dx + dy * dy - 0.02 * pr[3 * i + 2];
      if (d2 < bd) { bd = d2; best = i; }
    }
    if (best >= 0 && Math.sqrt(Math.max(bd, 0)) < 20) setSelected(best);
  };
  const containerRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const el = containerRef.current; if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const z = zoom * Math.exp(-e.deltaY * 0.0015);
      setZoom(Math.max(0.05, Math.min(50, z)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, setZoom]);

  // histogram handles
  const histValue = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect(); const W = ev.currentTarget.width, pad = 4;
    const x = (ev.clientX - rect.left) * W / rect.width;
    return hist.lo + (x - pad) / (W - 2 * pad) * (hist.hi - hist.lo);
  };
  const onHistDown = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const v = histValue(ev);
    histDragRef.current = Math.abs(v - vmin) < Math.abs(v - vmax) ? 1 : 2;
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onHistMove = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    if (!histDragRef.current) return;
    const v = histValue(ev);
    setRangeAuto(false);
    if (histDragRef.current === 1) setVmin(Math.min(v, vmax - 1e-9)); else setVmax(Math.max(v, vmin + 1e-9));
  };
  const onHistUp = (ev: React.PointerEvent<HTMLCanvasElement>) => { histDragRef.current = 0; ev.currentTarget.releasePointerCapture(ev.pointerId); model.save_changes(); };

  const toggleCategory = (code: number) => {
    const cur = new Set(hidden[channel] ?? []);
    if (cur.has(code)) cur.delete(code); else cur.add(code);
    setHidden({ ...hidden, [channel]: Array.from(cur).sort((a, b) => a - b) });
  };
  const setView = (normal: Vec3, up?: Vec3) => setRotation(viewMatrix(normal, up));

  // ---- layout --------------------------------------------------------------
  const label = (text: string) => <Typography sx={{ fontSize: 11, color: theme.textMuted, width: 78, flex: "none" }}>{text}</Typography>;
  const row = (text: string, child: React.ReactNode) => (
    <Stack direction="row" spacing={1} alignItems="center">{label(text)}{child}</Stack>
  );
  const sliderRow = (text: string, value: number, set: (v: number) => void, min: number, max: number, step: number, fmt = (v: number) => v.toFixed(2)) => row(text, (
    <>
      <Slider size="small" value={value} min={min} max={max} step={step} sx={{ ...sliderSx, flex: 1 }}
        onChange={(_, v) => set(v as number)} onChangeCommitted={() => model.save_changes()} />
      <Typography sx={{ fontSize: 11, width: 44, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmt(value)}</Typography>
    </>
  ));
  const switchRow = (text: string, value: boolean, set: (v: boolean) => void) => row(text, (
    <Switch size="small" checked={value} onChange={(_, v) => { set(v); model.save_changes(); }} />
  ));
  const selectSx = { fontSize: 12, height: 26, flex: 1, minWidth: 0 };

  return (
    <Box sx={{ display: "flex", gap: "10px", p: "6px", bgcolor: theme.bg, color: theme.text, borderRadius: 1, fontFamily: "system-ui, sans-serif" }}>
      <Stack spacing={0.5}>
        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>{title}</Typography>
        <Box ref={containerRef} sx={{ border: `1px solid ${theme.border}`, lineHeight: 0 }}>
          <canvas ref={canvasRef} data-quantem-scientific-output="atoms-3d"
            style={{ width: canvasSize, height: canvasSize, cursor: "grab", touchAction: "none" }}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
            onContextMenu={(e) => e.preventDefault()}
            onDoubleClick={() => { setZoom(1); setPan([0, 0]); model.save_changes(); }} />
        </Box>
        <Typography sx={{ fontSize: 10, color: theme.textMuted, minHeight: 14, maxWidth: canvasSize, whiteSpace: "normal" }}>
          {selectedInfo || "drag: rotate  |  scroll: zoom  |  shift-drag: pan  |  click: pick  |  double-click: reset"}
        </Typography>
      </Stack>

      <Stack spacing={0.5} sx={{ width: 260 }}>
        {row("channel", (
          <Select size="small" value={channel} sx={selectSx} onChange={(e) => { setChannel(String(e.target.value)); model.save_changes(); }}>
            {channelNames.map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
          </Select>
        ))}
        {row("colormap", (
          <Select size="small" value={cmap} sx={selectSx} disabled={isCategorical} onChange={(e) => { setCmap(String(e.target.value)); model.save_changes(); }}>
            {cmapNames.map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
          </Select>
        ))}
        <canvas ref={histRef} width={252} height={96}
          style={{ border: `1px solid ${theme.border}`, touchAction: "none", cursor: "col-resize" }}
          onPointerDown={onHistDown} onPointerMove={onHistMove} onPointerUp={onHistUp} />
        {row("range", (
          <>
            <input type="number" step="any" value={Number(vmin.toPrecision(5))} style={{ width: 76, fontSize: 11 }}
              onChange={(e) => { setRangeAuto(false); setVmin(Number(e.target.value)); model.save_changes(); }} />
            <input type="number" step="any" value={Number(vmax.toPrecision(5))} style={{ width: 76, fontSize: 11 }}
              onChange={(e) => { setRangeAuto(false); setVmax(Number(e.target.value)); model.save_changes(); }} />
            <Button size="small" sx={compactButton} onClick={() => { setRangeAuto(true); autoRange(); model.save_changes(); }}>auto</Button>
          </>
        ))}
        {switchRow("hide outside", hideOutside, setHideOutside)}
        {isCategorical && (
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: "2px 10px", maxHeight: 120, overflowY: "auto" }}>
            {[...(categories[channel] ?? []).map((lab, i) => ({ code: i, lab })), { code: -1, lab: "none" }].map(({ code, lab }) => {
              const [r, g, b] = colorOf(code);
              const off = hiddenCodes.has(code);
              return (
                <Box key={code} onClick={() => { toggleCategory(code); model.save_changes(); }}
                  sx={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer", fontSize: 11, opacity: off ? 0.35 : 1, textDecoration: off ? "line-through" : "none" }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: rgbString(r, g, b), border: "1px solid #555", display: "inline-block" }} />
                  {code}: {lab}
                </Box>
              );
            })}
          </Box>
        )}

        <Typography sx={{ fontSize: 11, fontWeight: 600, borderBottom: `1px solid ${theme.border}`, mt: 0.5 }}>clip</Typography>
        {switchRow("enable", clipEnabled, setClipEnabled)}
        {row("axis", (
          <Select size="small" value={clipAxis} sx={selectSx} onChange={(e) => { setClipAxis(String(e.target.value)); model.save_changes(); }}>
            {["view", "x", "y", "z", "custom"].map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
          </Select>
        ))}
        {sliderRow("center", clipCenter, setClipCenter, -modelRadius, modelRadius, modelRadius / 400)}
        {sliderRow("thickness", clipThickness, setClipThickness, 0.1, 2 * modelRadius, modelRadius / 400)}
        {row("normal", (
          <Typography sx={{ fontSize: 11, fontFamily: "monospace" }}>
            ({clipN[0].toFixed(2)}, {clipN[1].toFixed(2)}, {clipN[2].toFixed(2)})
          </Typography>
        ))}
        {switchRow("show plane", showClipPlane, setShowClipPlane)}
        {switchRow("show bounds", showBounds, setShowBounds)}

        <Typography sx={{ fontSize: 11, fontWeight: 600, borderBottom: `1px solid ${theme.border}`, mt: 0.5 }}>growth</Typography>
        {row("channel", (
          <Select size="small" value={growthChannel} sx={selectSx} displayEmpty
            onChange={(e) => {
              const name = String(e.target.value);
              setGrowthChannel(name);
              const k = channelNames.indexOf(name);
              if (k >= 0) { let hi = -Infinity; const v = channelData.subarray(k * numSites, (k + 1) * numSites); for (let i = 0; i < v.length; i++) if (v[i] > hi) hi = v[i]; setGrowthMax(hi); }
              model.save_changes();
            }}>
            <MenuItem value="" sx={{ fontSize: 12 }}>off</MenuItem>
            {channelNames.map((n) => <MenuItem key={n} value={n} sx={{ fontSize: 12 }}>{n}</MenuItem>)}
          </Select>
        ))}
        {growthValues && sliderRow("step", growthMax, setGrowthMax, growthRange.lo, growthRange.hi, growthRange.step, (v) => growthRange.step === 1 ? String(Math.round(v)) : v.toFixed(2))}

        <Typography sx={{ fontSize: 11, fontWeight: 600, borderBottom: `1px solid ${theme.border}`, mt: 0.5 }}>display</Typography>
        {sliderRow("marker", markerSize, setMarkerSize, 0.05, 2.0, 0.01)}
        {sliderRow("depth cue", depthCue, setDepthCue, 0, 1, 0.01)}
        {switchRow("edges", showEdges, setShowEdges)}
        {switchRow("axes", showAxes, setShowAxes)}
        {switchRow("dark", darkBackground, setDarkBackground)}
        {row("view", (
          <Stack direction="row" spacing={0.5}>
            <Button size="small" sx={compactButton} onClick={() => { setView([1, 0, 0], [0, 0, 1]); model.save_changes(); }}>x</Button>
            <Button size="small" sx={compactButton} onClick={() => { setView([0, 1, 0], [1, 0, 0]); model.save_changes(); }}>y</Button>
            <Button size="small" sx={compactButton} onClick={() => { setView([0, 0, 1], [0, 1, 0]); model.save_changes(); }}>z</Button>
            <Button size="small" sx={compactButton} onClick={() => { setRotation([...IDENTITY]); setZoom(1); setPan([0, 0]); model.save_changes(); }}>reset</Button>
          </Stack>
        ))}

        <Typography sx={{ fontSize: 11, fontWeight: 600, borderBottom: `1px solid ${theme.border}`, mt: 0.5 }}>site inspector</Typography>
        {switchRow("show", showInspector, setShowInspector)}
        {showInspector && (
          <>
            <canvas ref={nbRef} width={NEIGHBOR_CANVAS} height={NEIGHBOR_CANVAS}
              style={{ border: `1px solid ${theme.border}`, cursor: "grab", touchAction: "none" }}
              onPointerDown={onNbDown} onPointerMove={onNbMove} onPointerUp={onNbUp} />
            <Typography sx={{ fontSize: 10, color: theme.textMuted }}>
              {neighborhoodLabel || "neighbors (grey) and matched template (red +); drag to rotate"}
              {units ? `  [${units}]` : ""}
            </Typography>
          </>
        )}
      </Stack>
    </Box>
  );
}

export const render = createRender(ShowAtoms3D);
