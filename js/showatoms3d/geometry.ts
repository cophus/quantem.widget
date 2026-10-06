/**
 * Geometry helpers for ShowAtoms3D: 3x3 view matrices, slab clipping, and the
 * polygon where a plane cuts the model's bounding box.  Pure functions, no DOM.
 *
 * Conventions: a view matrix is row-major [right; up; normal] so that
 * (x, y, z) -> (u, v, depth) = M * p, with depth positive toward the viewer.
 */

export type Mat3 = number[]; // 9 numbers, row-major
export type Vec3 = [number, number, number];

export const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mat3mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Array<number>(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    }
  }
  return o;
}

export function mat3apply(m: Mat3, p: Vec3): Vec3 {
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
    m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
    m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
  ];
}

export function rotX(t: number): Mat3 {
  const c = Math.cos(t), s = Math.sin(t);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}

export function rotY(t: number): Mat3 {
  const c = Math.cos(t), s = Math.sin(t);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}

export function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Re-orthonormalize a drifting rotation matrix (Gram-Schmidt on the rows). */
export function orthonormalize(m: Mat3): Mat3 {
  const r = normalize([m[0], m[1], m[2]]);
  const u0: Vec3 = [m[3], m[4], m[5]];
  const d = dot(u0, r);
  const u = normalize([u0[0] - d * r[0], u0[1] - d * r[1], u0[2] - d * r[2]]);
  const n = cross(r, u);
  return [...r, ...u, ...n];
}

/** View matrix looking along `normal` (toward the viewer) with an approximate `up`. */
export function viewMatrix(normal: Vec3, up?: Vec3): Mat3 {
  const n = normalize(normal);
  let u0: Vec3 = up ? up : Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const d = dot(u0, n);
  const u = normalize([u0[0] - d * n[0], u0[1] - d * n[1], u0[2] - d * n[2]]);
  const r = cross(u, n);
  return [...r, ...u, ...n];
}

/** Rotate a unit normal by a screen-space drag (dx, dy) given the view rows. */
export function dragNormal(normal: Vec3, view: Mat3, dx: number, dy: number, gain: number): Vec3 {
  const right: Vec3 = [view[0], view[1], view[2]];
  const up: Vec3 = [view[3], view[4], view[5]];
  return normalize([
    normal[0] + gain * (dx * right[0] - dy * up[0]),
    normal[1] + gain * (dx * right[1] - dy * up[1]),
    normal[2] + gain * (dx * right[2] - dy * up[2]),
  ]);
}

/**
 * The clip normal in model coordinates for a clip axis.  For "view" it is the
 * current viewing direction; for x/y/z the model axis; otherwise `custom`.
 */
export function clipNormal(axis: string, view: Mat3, custom: Vec3): Vec3 {
  if (axis === "view") return [view[6], view[7], view[8]];
  if (axis === "x") return [1, 0, 0];
  if (axis === "y") return [0, 1, 0];
  if (axis === "z") return [0, 0, 1];
  return normalize(custom);
}

/**
 * Polygon (ordered corners) where the plane `n . p = d` cuts the axis-aligned
 * box [lo, hi].  Returns [] when the plane misses the box.
 */
export function planeBoxPolygon(n: Vec3, d: number, lo: Vec3, hi: Vec3): Vec3[] {
  const corners: Vec3[] = [];
  for (let i = 0; i < 8; i++) {
    corners.push([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]);
  }
  const edges: [number, number][] = [
    [0, 1], [2, 3], [4, 5], [6, 7],
    [0, 2], [1, 3], [4, 6], [5, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const pts: Vec3[] = [];
  for (const [a, b] of edges) {
    const pa = corners[a], pb = corners[b];
    const fa = dot(n, pa) - d, fb = dot(n, pb) - d;
    if ((fa < 0 && fb >= 0) || (fa >= 0 && fb < 0)) {
      const t = fa / (fa - fb);
      pts.push([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])]);
    }
  }
  if (pts.length < 3) return [];
  const c: Vec3 = [0, 0, 0];
  for (const p of pts) { c[0] += p[0] / pts.length; c[1] += p[1] / pts.length; c[2] += p[2] / pts.length; }
  // order by angle in the plane
  const e1 = normalize(cross(n, Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const e2 = cross(n, e1);
  return pts
    .map((p) => {
      const q: Vec3 = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
      return { p, ang: Math.atan2(dot(q, e2), dot(q, e1)) };
    })
    .sort((a, b) => a.ang - b.ang)
    .map((x) => x.p);
}

export interface ProjectOptions {
  view: Mat3;
  scale: number;   // px per model unit
  cx: number;      // screen center x (px)
  cy: number;      // screen center y (px)
  clip?: { normal: Vec3; center: number; thickness: number } | null;
}

/**
 * Project N packed xyz positions to screen space and evaluate slab clipping.
 * Writes (sx, sy, depth) into `out` and 1/0 into `visible`; returns the count.
 */
export function projectPositions(
  pos: Float32Array,
  opts: ProjectOptions,
  out: Float32Array,
  visible: Uint8Array,
): { count: number; dmin: number; dmax: number } {
  const m = opts.view;
  const n = pos.length / 3;
  let count = 0, dmin = Infinity, dmax = -Infinity;
  const clip = opts.clip ?? null;
  for (let i = 0; i < n; i++) {
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const u = m[0] * x + m[1] * y + m[2] * z;
    const v = m[3] * x + m[4] * y + m[5] * z;
    const dpt = m[6] * x + m[7] * y + m[8] * z;
    out[3 * i] = opts.cx + u * opts.scale;
    out[3 * i + 1] = opts.cy - v * opts.scale;
    out[3 * i + 2] = dpt;
    let ok = 1;
    if (clip) {
      const c = clip.normal[0] * x + clip.normal[1] * y + clip.normal[2] * z;
      if (Math.abs(c - clip.center) > clip.thickness / 2) ok = 0;
    }
    visible[i] = ok;
    if (ok) { count++; if (dpt < dmin) dmin = dpt; if (dpt > dmax) dmax = dpt; }
  }
  return { count, dmin, dmax };
}
