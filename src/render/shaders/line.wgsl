// Line shader (Parts B, C, D) — per-vertex coloured 3D lines [x, y, z, r, g, b].
// Prepended with colour.wgsl by createShader(), so toOutput() is available.

struct Uniforms {
  mvp : mat4x4f,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VSOut {
  @builtin(position) pos    : vec4f,
  @location(0)       colour : vec3f,
};

@vertex
fn vs(
  @location(0) position : vec3f,
  @location(1) colour   : vec3f,
) -> VSOut {
  var out : VSOut;
  out.pos = u.mvp * vec4f(position, 1.0);
  out.colour = colour;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  return toOutput(vec4f(in.colour, 1.0));
}
