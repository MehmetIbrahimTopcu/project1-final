# CS405 Project 1 — Transformations, Projection & Curves from Paintings

**Selected Painting:** Feyhaman Duran — *Kabataş Rıhtımı* (1936), Sakıp Sabancı Museum Collection (`kabatas`).

---

## 1. How to Run the Application

Requirements: **Node.js 20.19+** and a WebGPU-capable browser (Chrome, Edge, Safari 26+, or Firefox 141+).

```bash
npm install
npm run dev        # starts Vite dev server at http://localhost:5173
npm test           # runs all automated unit tests (Parts A, B, C, D)
npm run build      # verifies production bundle
```

---

## 2. Project Architecture (`src/`)

| Path | Description | Part |
|---|---|---|
| [`src/math/vec3.js`](src/math/vec3.js) | 3D vector library (`add`, `sub`, `scale`, `dot`, `cross`, `length`, `normalize`) | A, C |
| [`src/math/mat4.js`](src/math/mat4.js) | $4\times 4$ column-major `Float32Array(16)` library (`identity`, `multiply`, `translate`, `rotateX/Y/Z`, `scale`, `transpose`, `invertRigid`, `transformVec4`, `lookAt`, `perspective`, `orthographic`) | A, C |
| [`src/math/curves.js`](src/math/curves.js) | `bernstein3`, `cubicBezier`, `cubicBezierTangent`, `deCasteljau`, `c1NextControl`, `evalPiecewise`, `catmullRom` | D |
| [`src/math/quat.js`](src/math/quat.js) | Unit quaternion orientation and `slerp` camera interpolation | Bonus |
| [`src/scene/node.js`](src/scene/node.js) | `SceneNode` hierarchy with `updateWorld()` propagating $W_{\text{child}} = W_{\text{parent}} \cdot L_{\text{child}}$ | B |
| [`src/scene/curveExport.js`](src/scene/curveExport.js) | `curveToJSON()` and `downloadJSON()` exporting normalised painting coordinates $(u, v \in [0, 1])$ | D |
| [`src/scene/camera.js`](src/scene/camera.js) | Camera state and `viewProjection()` ($P \cdot V$) | C |
| [`src/scene/scene.js`](src/scene/scene.js) | Scene implementation for Parts A, B, C, D, and all 5 Bonus extensions | A–E, Bonus |
| [`src/render/meshRenderer.js`](src/render/meshRenderer.js) | 3D lit mesh pipeline with per-draw uniform buffer pool | A, B, D |
| [`src/render/lineRenderer.js`](src/render/lineRenderer.js) | 3D coloured line pipeline for local coordinate axes, control polygons, de Casteljau subdivision, and camera frustum | B, C, D |
| [`src/render/curveCompute.js`](src/render/curveCompute.js) | WebGPU `@compute` shader pipeline evaluating the curve on the GPU into a tessellated ribbon VBO | Bonus |
| [`src/main.js`](src/main.js) | Entry point, render loop, and pointer/UI wiring | C, D, E |

---

## 3. Controls & User Interface Guide (Part E & Bonus)

### Canvas Mouse Controls
- **Left-Drag on a Curve Control Point ($z = 0$):** Picks and drags that control point directly in the plane of the painting while automatically enforcing $C^1$ continuity at interior joints ($P_3$ and $P_6$).
- **Left-Drag in Freehand Trace Mode (`1. freehand trace & least-squares fit` ON):** Draws a freehand stroke on the painting ($z = 0$) and fits a 3-segment $C^1$ piecewise cubic Bézier curve via least-squares when released.
- **Left-Drag on Empty Space:** Orbits the active camera (`azimuth` and `elevation`) around `target`.
- **Shift + Left-Drag (or Middle-Drag):** Pans the active camera's `target`.
- **Mouse Wheel:** Zooms the active camera (`distance` in perspective mode, `orthoHeight` in orthographic mode).

### Control Panel Folders
1. **Painting:** Switch between the bundled SSM paintings (defaults to *Feyhaman Duran — Kabataş Rıhtımı*).
2. **Camera:**
   - **projection:** Switch at runtime between `perspective` and `orthographic` projection.
   - **active view:** Switch between `Main Camera` and `Inspection Camera (Frustum)`.
   - **Sliders:** Adjust `azimuth (°)`, `elevation (°)`, `distance`, `fovY (°)`, `near`, and `far`.
3. **Part A · Transformations:**
   - Controls the primary 3D object (**Harbour Mooring Bollard**) in front of the painting ($z > 0$), alongside two additional 3D objects.
   - **Order: `T·R·S` (on) / `S·R·T` (off):** Switches between $M = T \cdot R \cdot S$ and $M = S \cdot R \cdot T$.
   - **Show ghost of other order:** Draws a wireframe cube of the opposite multiplication order.
   - **Readout:** Displays both $T \cdot R \cdot S$ and $S \cdot R \cdot T$ $4\times 4$ matrices numerically in real time.
4. **Part B · Scene Graph:**
   - Implements a **5-level, 6-node hierarchy** (`boatHull` $\to$ `{ deckCabin, mainMast` $\to$ `boomArm` $\to$ `gaffSpar` $\to$ `pennantFlag }`).
   - **debug axes:** Draws RGB ($X, Y, Z$) local coordinate axes at every `SceneNode`.
5. **Part D · Curves:**
   - Traces the foreground sailboat hull and rigging arc in *Kabataş Rıhtımı* with a **3-segment (10 control points) piecewise $C^1$ cubic Bézier curve**.
   - **curve family:** Switch between `Piecewise C¹ Cubic Bézier` and `Catmull–Rom Spline`.
   - **de Casteljau `t`:** Adjusts $t \in [0, 1]$ and visualises the intermediate subdivision levels and evaluated point $B(t)$.
   - **animate object along curve:** Animates a 3D cube along the curve oriented by its tangent vector.
   - **export curve JSON:** Downloads `kabatas-curve.json` in normalised painting coordinates ($u, v \in [0, 1]$, origin top-left).
6. **Bonus · Extensions (+10):**
   - **1. freehand trace & least-squares fit:** Trace any contour on the painting with the mouse to fit a 3-segment $C^1$ cubic Bézier curve via chord-length parameterization and least squares.
   - **2. surface of revolution:** Revolves the curve profile around a vertical axis into a lit 3D mesh.
   - **3. painted perspective horizon & rays:** Displays the Bosphorus horizon line ($y = +0.18$) and vanishing rays converging at $\mathbf{V}_{\text{quay}} = (-0.18, +0.18, 0)$.
   - **4. quaternion orbit camera:** The view buttons move the camera with quaternion SLERP (`src/math/quat.js`).
   - **5. compute shader tessellated ribbon:** Evaluates the curve in a WebGPU `@compute` shader (`src/render/curveCompute.js`) and renders the resulting GPU buffer as a tessellated 3D ribbon.
7. **View:**
   - **Viewpoint buttons:** `view 1: front painting`, `view 2: angled 3D view`, `view 3: inspection camera`, `view 4 (bonus): match painter perspective`, and `reset`.
