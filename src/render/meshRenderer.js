import meshWGSL from './shaders/mesh.wgsl?raw';
import { createShader, fragmentConstants } from './shader.js';
import { DEPTH_FORMAT } from './renderer.js';
import { cube } from '../scene/geometry.js';
import * as mat4 from '../math/mat4.js';

const UNIFORM_FLOATS = 36;
const UNIFORM_BYTES = 144;

export class MeshRenderer {
  constructor(gpu) {
    const { device } = gpu;
    this.gpu = gpu;
    this.slots = [];
    this.cursor = 0;

    const cubeData = cube();
    this.cubeVertexCount = cubeData.length / 6;
    this.cubeVBO = device.createBuffer({
      label: 'cube vertices',
      size: cubeData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.cubeVBO, 0, cubeData);

    const module = createShader(device, meshWGSL, 'mesh');
    this.pipeline = device.createRenderPipeline({
      label: 'mesh pipeline',
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 6 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format: gpu.viewFormat }],
        constants: fragmentConstants(gpu),
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });
  }

  beginFrame() {
    this.cursor = 0;
  }

  #acquireSlot() {
    if (this.cursor < this.slots.length) {
      return this.slots[this.cursor++];
    }
    const { device } = this.gpu;
    const buffer = device.createBuffer({
      label: `mesh uniform #${this.slots.length}`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const scratch = new Float32Array(UNIFORM_FLOATS);
    const slot = { buffer, bindGroup, scratch };
    this.slots.push(slot);
    this.cursor++;
    return slot;
  }

  drawCube(pass, viewProj, model, colour, renderer) {
    this.drawMesh(pass, this.cubeVBO, this.cubeVertexCount, viewProj, model, colour, renderer);
  }

  drawMesh(pass, vbo, vertexCount, viewProj, model, colour, renderer) {
    if (vertexCount <= 0) return;
    const slot = this.#acquireSlot();
    const mvp = mat4.multiply(viewProj, model);
    slot.scratch.set(mvp, 0);
    slot.scratch.set(model, 16);
    slot.scratch[32] = colour[0] ?? 0.8;
    slot.scratch[33] = colour[1] ?? 0.8;
    slot.scratch[34] = colour[2] ?? 0.8;
    slot.scratch[35] = colour[3] ?? 1.0;

    this.gpu.device.queue.writeBuffer(slot.buffer, 0, slot.scratch);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, slot.bindGroup);
    pass.setVertexBuffer(0, vbo);
    pass.draw(vertexCount);
    renderer.countDraw(vertexCount);
  }
}
