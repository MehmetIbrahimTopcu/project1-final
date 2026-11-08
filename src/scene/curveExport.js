/**
 * Curve export — Part D.
 *
 * Required JSON format (control points in NORMALISED painting coordinates:
 * u, v ∈ [0, 1], origin at the TOP-LEFT corner of the painting, v downwards,
 * the same as texture coordinates):
 *
 * {
 *   "painting": "kavunlu",
 *   "kind": "bezier-c1",                 // or "catmull-rom" / "bspline"
 *   "segments": [                          // for "bezier-c1": 4 points each
 *     [[0.12, 0.40], [0.18, 0.31], [0.27, 0.30], [0.33, 0.38]],
 *     ...
 *   ],
 *   "points": [[u, v], ...]                // for "catmull-rom" / "bspline"
 * }
 *
 * The painting quad spans x ∈ [−a, a], y ∈ [−1, 1] at z = 0 (a = image
 * width / height), so  u = (x + a) / (2a)  and  v = (1 − y) / 2.
 */

export function curveToJSON(curve) {
  const {
    painting = 'kabatas',
    aspect = 1,
    kind = 'bezier-c1',
    segments = [],
    points = [],
  } = curve;

  const a = aspect || 1;
  const round4 = (n) => Math.round(n * 10000) / 10000;
  const toUV = (p) => [
    round4((p[0] + a) / (2 * a)),
    round4((1 - p[1]) / 2),
  ];

  const out = {
    painting,
    kind,
  };

  if (segments && segments.length > 0) {
    out.segments = segments.map((seg) => seg.map((p) => toUV(p)));
  }

  if (points && points.length > 0) {
    out.points = points.map((p) => toUV(p));
  }

  return out;
}

/** Provided: save any object as a .json download in the browser. */
export function downloadJSON(obj, filename = 'curve.json') {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
