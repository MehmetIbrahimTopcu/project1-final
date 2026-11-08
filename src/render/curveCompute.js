const COMPUTE_WGSL = `
struct Params {
  numSegments : u32,
  numSamples  : u32,
  halfWidth   : f32,
  zOffset     : f32,
  points      : array<vec4f, 12>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> outVerts : array<f32>;

fn evalBezier(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  let b0 = u * u * u;
  let b1 = 3.0 * u * u * t;
  let b2 = 3.0 * u * t * t;
  let b3 = t * t * t;
  return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3;
}

fn evalTangent(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let u = 1.0 - t;
  let c0 = 3.0 * u * u;
  let c1 = 6.0 * u * t;
  let c2 = 3.0 * t * t;
  return c0 * (p1 - p0) + c1 * (p2 - p1) + c2 * (p3 - p2);
}

fn samplePointAndNormal(idx: u32) -> array<vec3f, 2> {
  let u = f32(idx) / f32(params.numSamples);
  let nSeg = f32(params.numSegments);
  let scaled = clamp(u * nSeg, 0.0, nSeg - 0.00001);
  let seg = u32(floor(scaled));
  let localT = scaled - f32(seg);
  let base = seg * 4u;

  let p0 = params.points[base + 0u].xyz;
  let p1 = params.points[base + 1u].xyz;
  let p2 = params.points[base + 2u].xyz;
  let p3 = params.points[base + 3u].xyz;

  let pos = evalBezier(p0, p1, p2, p3, localT);
  let tanVec = evalTangent(p0, p1, p2, p3, localT);
  let lenXY = max(length(tanVec.xy), 0.00001);
  let dir = tanVec.xy / lenXY;
  let perp = vec3f(-dir.y, dir.x, 0.0);
  return array<vec3f, 2>(vec3f(pos.x, pos.y, params.zOffset), perp);
}

fn writeVertex(vIdx: u32, pos: vec3f, norm: vec3f) {
  let base = vIdx * 6u;
  outVerts[base + 0u] = pos.x;
  outVerts[base + 1u] = pos.y;
  outVerts[base + 2u] = pos.z;
  outVerts[base + 3u] = norm.x;
  outVerts[base + 4u] = norm.y;
  outVerts[base + 5u] = norm.z;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numSamples) {
    return;
  }

  let s0 = samplePointAndNormal(i);
  let s1 = samplePointAndNormal(i + 1u);

  let p0L = s0[0] + params.halfWidth * s0[1];
  let p0R = s0[0] - params.halfWidth * s0[1];
  let p1L = s1[0] + params.halfWidth * s1[1];
  let p1R = s1[0] - params.halfWidth * s1[1];

  let upNorm = vec3f(0.0, 0.0, 1.0);
  let vBase = i * 6u;

  writeVertex(vBase + 0u, p0L, upNorm);
  writeVertex(vBase + 1u, p0R, upNorm);
  writeVertex(vBase + 2u, p1L, upNorm);
  writeVertex(vBase + 3u, p1L, upNorm);
  writeVertex(vBase + 4u, p0R, upNorm);
  writeVertex(vBase + 5u, p1R, upNorm);
}
`;

export class CurveComputeRibbon {
  constructor(gpu, numSamples = 128) {
    const { device } = gpu;
    this.device = device;
    this.numSamples = numSamples;
    this.vertexCount = numSamples * 6;

    this.paramFloats = new Float32Array(4 + 12 * 4);
    this.paramUint = new Uint32Array(this.paramFloats.buffer, 0, 2);

    this.paramBuffer = device.createBuffer({
      label: 'compute curve params',
      size: this.paramFloats.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.vbo = device.createBuffer({
      label: 'compute ribbon vbo',
      size: this.vertexCount * 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const module = device.createShaderModule({
      label: 'curve compute shader',
      code: COMPUTE_WGSL,
    });

    this.pipeline = device.createComputePipeline({
      label: 'curve compute pipeline',
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'cs',
      },
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: this.vbo } },
      ],
    });
  }

  evaluate(segments, halfWidth = 0.022, zOffset = 0.016) {
    const numSegs = Math.min(3, segments.length);
    this.paramUint[0] = numSegs;
    this.paramUint[1] = this.numSamples;
    this.paramFloats[2] = halfWidth;
    this.paramFloats[3] = zOffset;

    let offset = 4;
    for (let s = 0; s < 3; s++) {
      const seg = segments[Math.min(s, numSegs - 1)];
      for (let k = 0; k < 4; k++) {
        const pt = seg[k];
        this.paramFloats[offset + 0] = pt[0] ?? 0;
        this.paramFloats[offset + 1] = pt[1] ?? 0;
        this.paramFloats[offset + 2] = pt[2] ?? 0;
        this.paramFloats[offset + 3] = 1.0;
        offset += 4;
      }
    }

    this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramFloats);

    const encoder = this.device.createCommandEncoder({ label: 'curve compute encoder' });
    const pass = encoder.beginComputePass({ label: 'curve compute pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.numSamples / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
