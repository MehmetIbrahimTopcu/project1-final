/**
 * curves — Part D.
 *
 * Points are plain arrays of any dimension: [x, y] in the plane of the
 * painting, or [x, y, z] in world space. Every function must work for both
 * (the tests use 2D and 3D). Return new arrays; never modify the inputs.
 *
 * A piecewise cubic Bézier curve is an array of segments, each segment being
 * four control points [P0, P1, P2, P3]. Consecutive segments share a joint:
 * segments[i][3] equals segments[i + 1][0].
 *
 * Tests: tests/curves.test.js
 */
import { todo } from './todo.js';

/* ---------------------------------------------------------- Bézier ------ */

/** The four cubic Bernstein basis values [B0(t), B1(t), B2(t), B3(t)]. */
export function bernstein3(t) {
  const u = 1 - t;
  return [
    u * u * u,
    3 * u * u * t,
    3 * u * t * t,
    t * t * t,
  ];
}

/** Point on the cubic Bézier curve with control points p0..p3 at t ∈ [0, 1]. */
export function cubicBezier(p0, p1, p2, p3, t) {
  const [b0, b1, b2, b3] = bernstein3(t);
  return p0.map((_, i) => b0 * p0[i] + b1 * p1[i] + b2 * p2[i] + b3 * p3[i]);
}

/** First derivative dB/dt of the cubic Bézier at t (not normalised). */
export function cubicBezierTangent(p0, p1, p2, p3, t) {
  const u = 1 - t;
  const c0 = 3 * u * u;
  const c1 = 6 * u * t;
  const c2 = 3 * t * t;
  return p0.map((_, i) =>
    c0 * (p1[i] - p0[i]) +
    c1 * (p2[i] - p1[i]) +
    c2 * (p3[i] - p2[i])
  );
}

/**
 * de Casteljau's construction for any number of control points.
 * Returns every level, for drawing the construction:
 *   levels[0] = the input points (n points)
 *   levels[1] = n − 1 interpolated points
 *   …
 *   levels[n − 1] = [the point on the curve]
 */
export function deCasteljau(points, t) {
  const levels = [points.map((p) => [...p])];
  let current = levels[0];
  while (current.length > 1) {
    const next = [];
    for (let i = 0; i < current.length - 1; i++) {
      const a = current[i];
      const b = current[i + 1];
      next.push(a.map((val, d) => val + (b[d] - val) * t));
    }
    levels.push(next);
    current = next;
  }
  return levels;
}

/* ----------------------------------------------- piecewise, C¹ joints --- */

/**
 * Given the last two control points P2, P3 of one segment, return the P1 of
 * the NEXT segment that makes the joint at P3 C¹-continuous.
 */
export function c1NextControl(p2, p3) {
  return p3.map((val, i) => 2 * val - p2[i]);
}

/**
 * Evaluate a piecewise cubic Bézier at a global parameter u ∈ [0, 1] that
 * runs over the whole curve: with n segments, segment i covers
 * u ∈ [i/n, (i+1)/n]. u = 0 is the first P0, u = 1 the last P3.
 */
export function evalPiecewise(segments, u) {
  const n = segments.length;
  const clamped = Math.max(0, Math.min(1, u));
  const scaled = clamped * n;
  const idx = Math.min(n - 1, Math.floor(scaled));
  const localT = scaled - idx;
  const [p0, p1, p2, p3] = segments[idx];
  return cubicBezier(p0, p1, p2, p3, localT);
}

/* ---------------------------- second curve family: implement ONE -------- */
/* The tests skip whichever of the two you leave as a TODO, and fail only   */
/* if you implement neither.                                                */

/**
 * Uniform Catmull–Rom: the segment between p1 and p2 (passes through both),
 * t ∈ [0, 1], with tangents (p2 − p0)/2 at p1 and (p3 − p1)/2 at p2.
 */
export function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return p0.map((_, i) => {
    const a = p0[i], b = p1[i], c = p2[i], d = p3[i];
    return 0.5 * (
      (2 * b) +
      (-a + c) * t +
      (2 * a - 5 * b + 4 * c - d) * t2 +
      (-a + 3 * b - 3 * c + d) * t3
    );
  });
}

/** Uniform cubic B-spline basis values [N0(t), N1(t), N2(t), N3(t)]. */
export function bsplineBasis(t) { return todo('curves.bsplineBasis'); }

/** Point on the uniform cubic B-spline segment controlled by p0..p3. */
export function bspline(p0, p1, p2, p3, t) { return todo('curves.bspline'); }
