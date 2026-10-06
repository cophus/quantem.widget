import { describe, expect, it } from "vitest";
import {
  IDENTITY, clipNormal, dragNormal, mat3apply, mat3mul, orthonormalize, planeBoxPolygon,
  projectPositions, rotX, rotY, viewMatrix,
} from "./geometry";

describe("showatoms3d geometry", () => {
  it("multiplies and applies 3x3 matrices row-major", () => {
    const m = mat3mul(rotX(0.3), rotY(-0.7));
    const p = mat3apply(m, [1, 2, 3]);
    const q = mat3apply(rotX(0.3), mat3apply(rotY(-0.7), [1, 2, 3]));
    expect(p[0]).toBeCloseTo(q[0]); expect(p[1]).toBeCloseTo(q[1]); expect(p[2]).toBeCloseTo(q[2]);
  });

  it("orthonormalizes a drifting rotation", () => {
    const m = orthonormalize([1.01, 0.02, 0, 0.01, 0.98, 0.03, 0, 0, 1.05]);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const d = m[3 * i] * m[3 * j] + m[3 * i + 1] * m[3 * j + 1] + m[3 * i + 2] * m[3 * j + 2];
        expect(d).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
  });

  it("builds a right-handed view matrix looking along the normal", () => {
    const v = viewMatrix([0, 0, 1]);
    expect(v.slice(6)).toEqual([0, 0, 1]);
    const vx = viewMatrix([1, 0, 0], [0, 0, 1]);
    expect(vx.slice(6)).toEqual([1, 0, 0]);
    expect(vx.slice(3, 6)).toEqual([0, 0, 1]);
  });

  it("resolves clip normals per axis", () => {
    expect(clipNormal("x", IDENTITY, [0, 0, 1])).toEqual([1, 0, 0]);
    expect(clipNormal("view", viewMatrix([0, 1, 0]), [0, 0, 1])).toEqual([0, 1, 0]);
    const c = clipNormal("custom", IDENTITY, [0, 3, 4]);
    expect(c[1]).toBeCloseTo(0.6); expect(c[2]).toBeCloseTo(0.8);
  });

  it("drags a normal toward the screen-right direction", () => {
    const n = dragNormal([0, 0, 1], IDENTITY, 10, 0, 0.1);
    expect(n[0]).toBeGreaterThan(0.5);
    expect(Math.hypot(...n)).toBeCloseTo(1);
  });

  it("cuts a unit cube with z = 0.25 into a square", () => {
    const poly = planeBoxPolygon([0, 0, 1], 0.25, [-1, -1, -1], [1, 1, 1]);
    expect(poly.length).toBe(4);
    for (const p of poly) expect(p[2]).toBeCloseTo(0.25);
    expect(planeBoxPolygon([0, 0, 1], 5, [-1, -1, -1], [1, 1, 1])).toEqual([]);
  });

  it("projects positions and applies slab clipping", () => {
    const pos = new Float32Array([0, 0, 0, 0, 0, 3, 1, 0, 0]);
    const out = new Float32Array(9), vis = new Uint8Array(3);
    const r = projectPositions(pos, { view: IDENTITY, scale: 10, cx: 100, cy: 100, clip: { normal: [0, 0, 1], center: 0, thickness: 1 } }, out, vis);
    expect(r.count).toBe(2);
    expect(Array.from(vis)).toEqual([1, 0, 1]);
    expect(out[6]).toBeCloseTo(110); // x = 1 -> 10 px right of center
    expect(out[7]).toBeCloseTo(100);
  });
});
