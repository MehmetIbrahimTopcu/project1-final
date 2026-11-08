// Mesh shader (Part A, B, D) — lit 3D geometry in linear RGB.
// Prepended with colour.wgsl by createShader(), so toOutput() is available.

struct Uniforms {
  mvp    : mat4x4f,
  model  : mat4x4f,
  colour : vec4f,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VSOut {
  @builtin(position) pos    : vec4f,
  @location(0)       normal : vec3f,
  @location(1)       worldP : vec3f,
};

@vertex
fn vs(
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
) -> VSOut {
  var out : VSOut;
  out.pos = u.mvp * vec4f(position, 1.0);
  out.worldP = (u.model * vec4f(position, 1.0)).xyz;
  // Transform normal by cofactor matrix (proportional to inverse-transpose (M^-1)^T)
  // so non-uniformly scaled objects keep perpendicular normals (Lecture L1-Oct12 Slide 12).
  let c0 = u.model[0].xyz;
  let c1 = u.model[1].xyz;
  let c2 = u.model[2].xyz;
  let nMat = mat3x3f(cross(c1, c2), cross(c2, c0), cross(c0, c1));
  out.normal = nMat * normal;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let nLen = length(in.normal);
  let N = select(vec3f(0.0, 1.0, 0.0), in.normal / nLen, nLen > 1e-5);

  // Two directional lights (key light + soft fill) in linear RGB
  let L1 = normalize(vec3f(0.45, 0.80, 0.55));
  let L2 = normalize(vec3f(-0.60, 0.25, -0.40));
  let diff1 = max(dot(N, L1), 0.0);
  let diff2 = max(dot(N, L2), 0.0);

  let ambient = 0.22;
  let shade = ambient + 0.68 * diff1 + 0.18 * diff2;
  let linearRGB = u.colour.rgb * shade;
  return toOutput(vec4f(linearRGB, u.colour.a));
}
