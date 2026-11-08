import * as vec3 from './vec3.js';

export function identity() {
  return [0, 0, 0, 1];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

export function normalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < 1e-8) return [0, 0, 0, 1];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

export function conjugate(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function fromAxisAngle(axis, rad) {
  const n = vec3.normalize(axis);
  const half = 0.5 * rad;
  const s = Math.sin(half);
  return normalize([n[0] * s, n[1] * s, n[2] * s, Math.cos(half)]);
}

export function multiply(q1, q2) {
  const v1 = [q1[0], q1[1], q1[2]], w1 = q1[3];
  const v2 = [q2[0], q2[1], q2[2]], w2 = q2[3];
  const crossV = vec3.cross(v1, v2);
  const dotV = vec3.dot(v1, v2);
  return [
    w1 * v2[0] + w2 * v1[0] + crossV[0],
    w1 * v2[1] + w2 * v1[1] + crossV[1],
    w1 * v2[2] + w2 * v1[2] + crossV[2],
    w1 * w2 - dotV,
  ];
}

export function fromAzimuthElevation(azimuthRad, elevationRad) {
  const qY = fromAxisAngle([0, 1, 0], azimuthRad);
  const qX = fromAxisAngle([1, 0, 0], -elevationRad);
  return normalize(multiply(qY, qX));
}

export function rotateVec3(q, v) {
  const qv = [q[0], q[1], q[2]];
  const w = q[3];
  const t = vec3.scale(vec3.cross(qv, v), 2.0);
  const wt = vec3.scale(t, w);
  const qxt = vec3.cross(qv, t);
  return vec3.add(v, vec3.add(wt, qxt));
}

export function nlerp(q0, q1, t, shortestArc = true) {
  let b = q1;
  if (shortestArc && dot(q0, q1) < 0) {
    b = [-q1[0], -q1[1], -q1[2], -q1[3]];
  }
  const u = 1.0 - t;
  return normalize([
    u * q0[0] + t * b[0],
    u * q0[1] + t * b[1],
    u * q0[2] + t * b[2],
    u * q0[3] + t * b[3],
  ]);
}

export function slerp(q0, q1, t, shortestArc = true) {
  let b = q1;
  let cosOmega = dot(q0, q1);
  if (shortestArc && cosOmega < 0) {
    cosOmega = -cosOmega;
    b = [-q1[0], -q1[1], -q1[2], -q1[3]];
  }
  const clamped = Math.max(-1.0, Math.min(1.0, cosOmega));
  const omega = Math.acos(clamped);

  if (omega < 0.01) {
    return nlerp(q0, b, t, false);
  }

  const sinOmega = Math.sin(omega);
  const w0 = Math.sin((1.0 - t) * omega) / sinOmega;
  const w1 = Math.sin(t * omega) / sinOmega;
  return normalize([
    w0 * q0[0] + w1 * b[0],
    w0 * q0[1] + w1 * b[1],
    w0 * q0[2] + w1 * b[2],
    w0 * q0[3] + w1 * b[3],
  ]);
}
