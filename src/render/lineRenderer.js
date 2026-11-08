import lineWGSL from './shaders/line.wgsl?raw';
import { createShader, fragmentConstants } from './shader.js';
import { DEPTH_FORMAT } from './renderer.js';
import { axes } from '../scene/geometry.js';
import * as mat4 from '../math/mat4.js';

export class LineRenderer {
  constructor(gpu) {
    const { device } = gpu;
    this.gpu = gpu;
    this.axesSlots = [];
    this.axesCursor = 0;
    this.dynSlots = [];
    this.dynCursor = 0;

    const axesData = axes(1.0);
    this.axesVertexCount = axesData.length / 6;
    this.axesVBO = device.createBuffer({
      label: 'axes vertices',
      size: axesData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.axesVBO, 0, axesData);

    const module = createShader(device, lineWGSL, 'line');
    this.pipeline = device.createRenderPipeline({
      label: 'line pipeline',
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
      primitive: { topology: 'line-list' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: 'less-equal',
      },
    });
  }

  beginFrame() {
    this.axesCursor = 0;
    this.dynCursor = 0;
  }

  #acquireAxesSlot() {
    if (this.axesCursor < this.axesSlots.length) {
      return this.axesSlots[this.axesCursor++];
    }
    const { device } = this.gpu;
    const buffer = device.createBuffer({
      label: `axes uniform #${this.axesSlots.length}`,
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const slot = { buffer, bindGroup };
    this.axesSlots.push(slot);
    this.axesCursor++;
    return slot;
  }

  #acquireDynSlot(minBytes) {
    const { device } = this.gpu;
    if (this.dynCursor < this.dynSlots.length) {
      const slot = this.dynSlots[this.dynCursor++];
      if (slot.vboSize < minBytes) {
        slot.vbo.destroy();
        slot.vboSize = Math.max(minBytes, slot.vboSize * 2);
        slot.vbo = device.createBuffer({
          label: 'dynamic line vbo',
          size: slot.vboSize,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
      }
      return slot;
    }
    const ubo = device.createBuffer({
      label: `dyn line uniform #${this.dynSlots.length}`,
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });
    const vboSize = Math.max(4096, minBytes);
    const vbo = device.createBuffer({
      label: `dyn line vbo #${this.dynSlots.length}`,
      size: vboSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const slot = { ubo, bindGroup, vbo, vboSize };
    this.dynSlots.push(slot);
    this.dynCursor++;
    return slot;
  }

  drawAxes(pass, viewProj, world, len, renderer) {
    const slot = this.#acquireAxesSlot();
    const model = len === 1 ? world : mat4.multiply(world, mat4.scale(len, len, len));
    const mvp = mat4.multiply(viewProj, model);
    this.gpu.device.queue.writeBuffer(slot.buffer, 0, mvp);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, slot.bindGroup);
    pass.setVertexBuffer(0, this.axesVBO);
    pass.draw(this.axesVertexCount);
    renderer.countDraw(this.axesVertexCount);
  }

  drawLines(pass, mvp, vertices, renderer) {
    const vertexCount = Math.floor(vertices.length / 6);
    if (vertexCount < 2) return;
    const slot = this.#acquireDynSlot(vertices.byteLength);
    this.gpu.device.queue.writeBuffer(slot.ubo, 0, mvp);
    this.gpu.device.queue.writeBuffer(slot.vbo, 0, vertices);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, slot.bindGroup);
    pass.setVertexBuffer(0, slot.vbo);
    pass.draw(vertexCount);
    renderer.countDraw(vertexCount);
  }
}
