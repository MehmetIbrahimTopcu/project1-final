/**
 * YOUR SCENE — Parts A, B and D live here (and in files you add next to it).
 *
 * main.js calls, every frame:
 *     scene.update(dt, time)          dt and time in seconds
 *     scene.draw(pass, viewProj)      inside the frame's render pass
 * and scene.reset() from the panel's "reset" button.
 */

import * as vec3 from '../math/vec3.js';
import * as mat4 from '../math/mat4.js';
import * as curves from '../math/curves.js';
import * as quat from '../math/quat.js';
import { createCamera } from './camera.js';
import { SceneNode } from './node.js';
import { curveToJSON, downloadJSON } from './curveExport.js';
import { MeshRenderer } from '../render/meshRenderer.js';
import { LineRenderer } from '../render/lineRenderer.js';
import { CurveComputeRibbon } from '../render/curveCompute.js';

const mulAll = (...ms) => ms.reduce((acc, m) => mat4.multiply(acc, m));

function intersectRayZ0(cam, aspect, ndcX, ndcY) {
  const f = vec3.normalize(vec3.sub(cam.target, cam.eye));
  let r = vec3.cross(f, cam.up);
  if (vec3.length(r) < 1e-6) {
    const fallbackUp = Math.abs(f[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    r = vec3.cross(f, fallbackUp);
  }
  r = vec3.normalize(r);
  const u = vec3.cross(r, f);

  let rayOrigin;
  let rayDir;

  if (cam.projection === 'orthographic') {
    const halfH = cam.orthoHeight * 0.5;
    const halfW = halfH * aspect;
    rayOrigin = vec3.add(
      cam.eye,
      vec3.add(vec3.scale(r, ndcX * halfW), vec3.scale(u, ndcY * halfH))
    );
    rayDir = f;
  } else {
    const tanHalf = Math.tan(cam.fovY * 0.5);
    const px = ndcX * aspect * tanHalf;
    const py = ndcY * tanHalf;
    rayOrigin = cam.eye;
    rayDir = vec3.normalize(
      vec3.add(f, vec3.add(vec3.scale(r, px), vec3.scale(u, py)))
    );
  }

  if (Math.abs(rayDir[2]) < 1e-6) return null;
  const t = -rayOrigin[2] / rayDir[2];
  if (t < 0 && cam.projection === 'perspective') return null;
  return [
    rayOrigin[0] + t * rayDir[0],
    rayOrigin[1] + t * rayDir[1],
    0.0,
  ];
}

function createInitialControlPoints() {
  const P0 = [-0.33, -0.16, 0.0];
  const P1 = [-0.31, -0.29, 0.0];
  const P2 = [-0.16, -0.31, 0.0];
  const P3 = [-0.02, -0.31, 0.0];
  const P4 = curves.c1NextControl(P2, P3);
  const P5 = [ 0.29, -0.26, 0.0];
  const P6 = [ 0.41, -0.17, 0.0];
  const P7 = curves.c1NextControl(P5, P6);
  const P8 = [ 0.65, -0.03, 0.0];
  const P9 = [ 0.76,  0.01, 0.0];
  return [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9];
}

function fitCubicSegmentLeastSquares(samples) {
  const m = samples.length;
  const p0 = [...samples[0]];
  const p3 = [...samples[m - 1]];
  if (m === 2) {
    return [
      p0,
      vec3.add(vec3.scale(p0, 2 / 3), vec3.scale(p3, 1 / 3)),
      vec3.add(vec3.scale(p0, 1 / 3), vec3.scale(p3, 2 / 3)),
      p3,
    ];
  }

  const chords = [0];
  for (let i = 1; i < m; i++) {
    const d = Math.hypot(samples[i][0] - samples[i - 1][0], samples[i][1] - samples[i - 1][1]);
    chords.push(chords[i - 1] + d);
  }
  const totalLen = chords[m - 1];
  const tVals = totalLen > 1e-6
    ? chords.map((c) => c / totalLen)
    : samples.map((_, i) => i / (m - 1));

  let c11 = 0;
  let c12 = 0;
  let c22 = 0;
  let x1 = [0, 0, 0];
  let x2 = [0, 0, 0];

  for (let i = 0; i < m; i++) {
    const [b0, b1, b2, b3] = curves.bernstein3(tVals[i]);
    c11 += b1 * b1;
    c12 += b1 * b2;
    c22 += b2 * b2;

    const q = samples[i];
    const rx = q[0] - b0 * p0[0] - b3 * p3[0];
    const ry = q[1] - b0 * p0[1] - b3 * p3[1];
    x1[0] += b1 * rx;
    x1[1] += b1 * ry;
    x2[0] += b2 * rx;
    x2[1] += b2 * ry;
  }

  const det = c11 * c22 - c12 * c12;
  if (Math.abs(det) < 1e-8) {
    return [
      p0,
      vec3.add(vec3.scale(p0, 2 / 3), vec3.scale(p3, 1 / 3)),
      vec3.add(vec3.scale(p0, 1 / 3), vec3.scale(p3, 2 / 3)),
      p3,
    ];
  }

  const p1 = [
    (c22 * x1[0] - c12 * x2[0]) / det,
    (c22 * x1[1] - c12 * x2[1]) / det,
    0.0,
  ];
  const p2 = [
    (c11 * x2[0] - c12 * x1[0]) / det,
    (c11 * x2[1] - c12 * x1[1]) / det,
    0.0,
  ];
  return [p0, p1, p2, p3];
}

function fitPiecewiseBezierLeastSquares(stroke) {
  if (stroke.length < 4) return null;

  const arc = [0];
  for (let i = 1; i < stroke.length; i++) {
    const d = Math.hypot(stroke[i][0] - stroke[i - 1][0], stroke[i][1] - stroke[i - 1][1]);
    arc.push(arc[i - 1] + d);
  }
  const total = arc[arc.length - 1];
  if (total < 0.05) return null;

  const split1 = total / 3;
  const split2 = (2 * total) / 3;
  let i1 = stroke.findIndex((_p, idx) => arc[idx] >= split1);
  let i2 = stroke.findIndex((_p, idx) => arc[idx] >= split2);
  i1 = Math.max(1, Math.min(stroke.length - 3, i1));
  i2 = Math.max(i1 + 1, Math.min(stroke.length - 2, i2));

  const seg0 = fitCubicSegmentLeastSquares(stroke.slice(0, i1 + 1));
  const seg1 = fitCubicSegmentLeastSquares(stroke.slice(i1, i2 + 1));
  const seg2 = fitCubicSegmentLeastSquares(stroke.slice(i2));

  const P0 = seg0[0];
  const P1 = seg0[1];
  const P3 = seg0[3];
  const delta1 = [
    0.5 * ((P3[0] - seg0[2][0]) + (seg1[1][0] - P3[0])),
    0.5 * ((P3[1] - seg0[2][1]) + (seg1[1][1] - P3[1])),
    0.0,
  ];
  const P2 = [P3[0] - delta1[0], P3[1] - delta1[1], 0.0];
  const P4 = curves.c1NextControl(P2, P3);

  const P6 = seg1[3];
  const delta2 = [
    0.5 * ((P6[0] - seg1[2][0]) + (seg2[1][0] - P6[0])),
    0.5 * ((P6[1] - seg1[2][1]) + (seg2[1][1] - P6[1])),
    0.0,
  ];
  const P5 = [P6[0] - delta2[0], P6[1] - delta2[1], 0.0];
  const P7 = curves.c1NextControl(P5, P6);

  const P8 = seg2[2];
  const P9 = seg2[3];

  return [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9];
}

export function createScene({ gpu, renderer, painting, panel, camera = createCamera(), cf }) {
  const meshRenderer = new MeshRenderer(gpu);
  const lineRenderer = new LineRenderer(gpu);
  const computeRibbon = new CurveComputeRibbon(gpu, 128);

  let sweptVBO = gpu.device.createBuffer({
    label: 'swept curve mesh vbo',
    size: 64 * 1024,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  let sweptVBOSize = 64 * 1024;
  let sweptVertexCount = 0;

  const defaultState = () => ({
    tx: -0.88,
    ty: -0.58,
    tz: 0.24,
    rx: 0.20,
    ry: 0.45,
    rz: 0.15,
    sx: 0.12,
    sy: 0.26,
    sz: 0.12,
    useTRS: true,
    showGhostOrder: true,
    animateHierarchy: true,
    animSpeed: 1.0,
    hullRollDeg: 12,
    mastYawDeg: 18,
    boomAngleDeg: 28,
    gaffAngleDeg: -32,
    debugAxes: true,
    curveFamily: 'bezier',
    deCasteljauT: 0.38,
    showControlPolygon: true,
    showDeCasteljau: true,
    animateCurveObject: true,
    curveAnimSpeed: 0.22,
    freehandTraceMode: false,
    showSweptMesh: false,
    showPerspectiveOverlay: false,
    showComputeRibbon: false,
  });

  const defaultCamControl = () => ({
    viewMode: 'main',
    azimuthDeg: 0,
    elevationDeg: 0,
    distance: 2.8,
    fovDeg: 45,
    near: 0.2,
    far: 25.0,
    inspectAzimuthDeg: 38,
    inspectElevationDeg: 24,
    inspectDistance: 5.2,
  });

  const state = defaultState();
  const camCtrl = defaultCamControl();
  const mainCam = createCamera();
  const inspectionCamera = createCamera();
  inspectionCamera.near = 0.15;
  inspectionCamera.far = 50.0;
  inspectionCamera.fovY = (48 * Math.PI) / 180;

  let cameraQuat = quat.identity();
  let camTransition = null;
  let isDraggingPoint = false;
  let isTracingStroke = false;
  let tracedStroke = [];

  let controlPoints = createInitialControlPoints();
  let selectedPointIndex = -1;
  let curveRunnerU = 0.0;

  const boatHull = new SceneNode('boatHull');
  const deckCabin = new SceneNode('deckCabin');
  const mainMast = new SceneNode('mainMast');
  const boomArm = new SceneNode('boomArm');
  const gaffSpar = new SceneNode('gaffSpar');
  const pennantFlag = new SceneNode('pennantFlag');

  boatHull.addChild(deckCabin);
  boatHull.addChild(mainMast);
  mainMast.addChild(boomArm);
  boomArm.addChild(gaffSpar);
  gaffSpar.addChild(pennantFlag);

  const hierarchyNodes = [
    {
      node: boatHull,
      colour: [0.42, 0.23, 0.11, 1.0],
      meshOffset: [0.0, 0.0, 0.0],
      meshScale: [0.42, 0.10, 0.18],
      axesSize: 0.22,
    },
    {
      node: deckCabin,
      colour: [0.76, 0.64, 0.45, 1.0],
      meshOffset: [0.0, 0.04, 0.0],
      meshScale: [0.14, 0.08, 0.12],
      axesSize: 0.14,
    },
    {
      node: mainMast,
      colour: [0.58, 0.36, 0.16, 1.0],
      meshOffset: [0.0, 0.22, 0.0],
      meshScale: [0.04, 0.44, 0.04],
      axesSize: 0.18,
    },
    {
      node: boomArm,
      colour: [0.82, 0.52, 0.20, 1.0],
      meshOffset: [0.17, 0.0, 0.0],
      meshScale: [0.34, 0.035, 0.035],
      axesSize: 0.16,
    },
    {
      node: gaffSpar,
      colour: [0.90, 0.78, 0.58, 1.0],
      meshOffset: [0.13, 0.0, 0.0],
      meshScale: [0.26, 0.028, 0.028],
      axesSize: 0.14,
    },
    {
      node: pennantFlag,
      colour: [0.88, 0.16, 0.14, 1.0],
      meshOffset: [0.05, 0.0, 0.0],
      meshScale: [0.10, 0.05, 0.015],
      axesSize: 0.12,
    },
  ];

  let matTRS = mat4.identity();
  let matSRT = mat4.identity();
  let activePartAModel = mat4.identity();
  let alternatePartAModel = mat4.identity();
  let secondObjectModel = mat4.identity();
  let thirdObjectModel = mat4.identity();

  const uiControllers = [];
  const reg = (ctrl) => {
    if (ctrl && typeof ctrl.update === 'function') uiControllers.push(ctrl);
    return ctrl;
  };

  function syncCamerasFromControls() {
    const deg2rad = Math.PI / 180;
    const az = camCtrl.azimuthDeg * deg2rad;
    const el = Math.max(-85 * deg2rad, Math.min(85 * deg2rad, camCtrl.elevationDeg * deg2rad));
    const r = Math.max(0.4, camCtrl.distance);

    cameraQuat = quat.fromAzimuthElevation(az, el);
    const offset = quat.rotateVec3(cameraQuat, [0, 0, r]);
    mainCam.eye = vec3.add(mainCam.target, offset);
    mainCam.fovY = camCtrl.fovDeg * deg2rad;
    mainCam.near = Math.max(0.05, camCtrl.near);
    mainCam.far = Math.max(mainCam.near + 0.5, camCtrl.far);

    const iAz = camCtrl.inspectAzimuthDeg * deg2rad;
    const iEl = Math.max(-85 * deg2rad, Math.min(85 * deg2rad, camCtrl.inspectElevationDeg * deg2rad));
    const iR = Math.max(1.2, camCtrl.inspectDistance);
    const iQuat = quat.fromAzimuthElevation(iAz, iEl);
    inspectionCamera.target = [0.0, -0.1, 0.8];
    inspectionCamera.eye = vec3.add(inspectionCamera.target, quat.rotateVec3(iQuat, [0, 0, iR]));

    applyActiveCamera();
  }

  function applyActiveCamera() {
    const src = camCtrl.viewMode === 'inspection' ? inspectionCamera : mainCam;
    camera.eye = [...src.eye];
    camera.target = [...src.target];
    camera.up = [...src.up];
    camera.fovY = src.fovY;
    camera.near = src.near;
    camera.far = src.far;
    camera.orthoHeight = src.orthoHeight;
    if (camCtrl.viewMode === 'inspection') {
      camera.projection = 'perspective';
    } else {
      camera.projection = mainCam.projection;
    }
  }

  function startCameraTransition({ azimuthDeg, elevationDeg, distance, fovDeg, target, duration = 0.85 }) {
    const deg2rad = Math.PI / 180;
    const qStart = [...cameraQuat];
    const qEnd = quat.fromAzimuthElevation(azimuthDeg * deg2rad, elevationDeg * deg2rad);
    camTransition = {
      elapsed: 0,
      duration,
      qStart,
      qEnd,
      azEnd: azimuthDeg,
      elEnd: elevationDeg,
      distStart: camCtrl.distance,
      distEnd: distance,
      fovStart: camCtrl.fovDeg,
      fovEnd: fovDeg ?? camCtrl.fovDeg,
      targetStart: [...mainCam.target],
      targetEnd: [...target],
    };
  }

  let viewModeSelect = null;
  if (cf) {
    viewModeSelect = cf.select(
      'active view',
      [
        { value: 'main', label: 'Main Camera' },
        { value: 'inspection', label: 'Inspection Camera (Frustum)' },
      ],
      camCtrl.viewMode,
      (v) => {
        if (camCtrl.viewMode === 'main') mainCam.projection = camera.projection;
        camCtrl.viewMode = v;
        syncCamerasFromControls();
      }
    );
    reg(cf.slider('azimuth (°)', camCtrl, 'azimuthDeg', -180, 180, 1, () => { camTransition = null; syncCamerasFromControls(); }));
    reg(cf.slider('elevation (°)', camCtrl, 'elevationDeg', -85, 85, 1, () => { camTransition = null; syncCamerasFromControls(); }));
    reg(cf.slider('distance', camCtrl, 'distance', 0.6, 8.0, 0.05, () => { camTransition = null; syncCamerasFromControls(); }));
    reg(cf.slider('fovY (°)', camCtrl, 'fovDeg', 15, 100, 1, () => { camTransition = null; syncCamerasFromControls(); }));
    reg(cf.slider('near', camCtrl, 'near', 0.05, 2.0, 0.05, syncCamerasFromControls));
    reg(cf.slider('far', camCtrl, 'far', 2.0, 100.0, 0.5, syncCamerasFromControls));
  }
  syncCamerasFromControls();

  const folderA = panel.folder('Part A · Transformations');
  reg(folderA.toggle('Order: T·R·S (on) / S·R·T (off)', state, 'useTRS'));
  reg(folderA.toggle('Show ghost of other order', state, 'showGhostOrder'));
  reg(folderA.slider('translate X', state, 'tx', -1.2, 1.2, 0.01));
  reg(folderA.slider('translate Y', state, 'ty', -1.0, 1.0, 0.01));
  reg(folderA.slider('translate Z', state, 'tz', 0.08, 1.2, 0.01));
  reg(folderA.slider('rotate X (rad)', state, 'rx', -Math.PI, Math.PI, 0.01));
  reg(folderA.slider('rotate Y (rad)', state, 'ry', -Math.PI, Math.PI, 0.01));
  reg(folderA.slider('rotate Z (rad)', state, 'rz', -Math.PI, Math.PI, 0.01));
  reg(folderA.slider('scale X', state, 'sx', 0.05, 0.8, 0.01));
  reg(folderA.slider('scale Y', state, 'sy', 0.05, 0.8, 0.01));
  reg(folderA.slider('scale Z', state, 'sz', 0.05, 0.8, 0.01));
  const readoutA = folderA.text('');

  const folderB = panel.folder('Part B · Scene Graph');
  reg(folderB.toggle('debug axes', state, 'debugAxes'));
  reg(folderB.toggle('animate', state, 'animateHierarchy'));
  reg(folderB.slider('anim speed', state, 'animSpeed', 0.0, 2.5, 0.05));
  reg(folderB.slider('Hull roll (deg)', state, 'hullRollDeg', -35, 35, 1));
  reg(folderB.slider('Mast yaw (deg)', state, 'mastYawDeg', -90, 90, 1));
  reg(folderB.slider('Boom shoulder (deg)', state, 'boomAngleDeg', -85, 85, 1));
  reg(folderB.slider('Gaff elbow (deg)', state, 'gaffAngleDeg', -120, 120, 1));

  const folderD = panel.folder('Part D · Curves');
  const familySelect = folderD.select(
    'curve family',
    [
      { value: 'bezier', label: 'Piecewise C¹ Cubic Bézier' },
      { value: 'catmull-rom', label: 'Catmull–Rom Spline' },
    ],
    state.curveFamily,
    (v) => { state.curveFamily = v; }
  );
  reg(folderD.slider('de Casteljau t', state, 'deCasteljauT', 0.0, 1.0, 0.005));
  reg(folderD.toggle('show control polygon', state, 'showControlPolygon'));
  reg(folderD.toggle('show de Casteljau', state, 'showDeCasteljau'));
  reg(folderD.toggle('animate object along curve', state, 'animateCurveObject'));
  reg(folderD.slider('curve speed', state, 'curveAnimSpeed', 0.05, 0.8, 0.01));
  folderD.button('export curve JSON', () => {
    downloadJSON(getCurveJSON(), 'kabatas-curve.json');
  });

  const folderBonus = panel.folder('Bonus · Extensions (+10)');
  reg(folderBonus.toggle('1. freehand trace & least-squares fit', state, 'freehandTraceMode'));
  reg(folderBonus.toggle('2. surface of revolution', state, 'showSweptMesh'));
  reg(folderBonus.toggle('3. painted perspective horizon & rays', state, 'showPerspectiveOverlay'));
  reg(folderBonus.toggle('5. compute shader tessellated ribbon', state, 'showComputeRibbon'));

  function setupViewFolder(vf) {
    vf.button('view 1: front painting', () => {
      camCtrl.viewMode = 'main';
      viewModeSelect?.set('main');
      startCameraTransition({
        azimuthDeg: 0,
        elevationDeg: 0,
        distance: 2.8,
        fovDeg: 45,
        target: [0, 0, 0],
      });
    });
    vf.button('view 2: angled 3D view', () => {
      camCtrl.viewMode = 'main';
      viewModeSelect?.set('main');
      startCameraTransition({
        azimuthDeg: 28,
        elevationDeg: 16,
        distance: 2.75,
        fovDeg: 45,
        target: [0, -0.15, 0.25],
      });
    });
    vf.button('view 3: inspection camera', () => {
      camTransition = null;
      if (camCtrl.viewMode === 'main') mainCam.projection = camera.projection;
      camCtrl.viewMode = 'inspection';
      viewModeSelect?.set('inspection');
      syncCamerasFromControls();
    });
    vf.button('view 4 (bonus): match painter perspective', () => {
      camCtrl.viewMode = 'main';
      mainCam.projection = 'perspective';
      camera.projection = 'perspective';
      viewModeSelect?.set('main');
      state.showPerspectiveOverlay = true;
      for (const ctrl of uiControllers) ctrl.update();
      startCameraTransition({
        azimuthDeg: -11,
        elevationDeg: 8,
        distance: 2.95,
        fovDeg: 38,
        target: [-0.05, -0.06, 0.12],
      });
    });
  }

  function getBezierSegments() {
    const segments = [];
    for (let i = 0; i + 3 < controlPoints.length; i += 3) {
      segments.push([
        controlPoints[i],
        controlPoints[i + 1],
        controlPoints[i + 2],
        controlPoints[i + 3],
      ]);
    }
    return segments;
  }

  function getCurveJSON() {
    const aspect = painting.aspect || (1280 / 1026);
    const kind = state.curveFamily === 'bezier' ? 'bezier-c1' : state.curveFamily;
    return curveToJSON({
      painting: painting.id || 'kabatas',
      kind,
      aspect,
      segments: getBezierSegments(),
      points: controlPoints,
    });
  }

  function evalActiveCurve(u, family = state.curveFamily) {
    const clampedU = Math.max(0, Math.min(1, u));
    if (family === 'catmull-rom') {
      const numSegs = controlPoints.length - 3;
      const s = clampedU * numSegs;
      const seg = Math.min(Math.floor(s), numSegs - 1);
      const localT = s - seg;
      const p0 = controlPoints[seg];
      const p1 = controlPoints[seg + 1];
      const p2 = controlPoints[seg + 2];
      const p3 = controlPoints[seg + 3];
      const eps = 1e-3;
      const t0 = Math.max(0, localT - eps);
      const t1 = Math.min(1, localT + eps);
      const dt = Math.max(1e-6, t1 - t0);
      const a = curves.catmullRom(p0, p1, p2, p3, t0);
      const b = curves.catmullRom(p0, p1, p2, p3, t1);
      return {
        point: curves.catmullRom(p0, p1, p2, p3, localT),
        tangent: vec3.scale(vec3.sub(b, a), 1 / dt),
      };
    }
    const segs = getBezierSegments();
    const n = segs.length;
    const scaled = clampedU * n;
    const idx = Math.min(n - 1, Math.floor(scaled));
    const localT = scaled - idx;
    const [p0, p1, p2, p3] = segs[idx];
    return {
      point: curves.evalPiecewise(segs, clampedU),
      tangent: curves.cubicBezierTangent(p0, p1, p2, p3, localT),
    };
  }

  function updateSweptMeshVBO() {
    const N = 40;
    const verts = [];
    const pushTri = (a, na, b, nb, c, nc) => {
      verts.push(...a, ...na, ...b, ...nb, ...c, ...nc);
    };

    const M = 20;
    const axisX = -0.15;
    const zCenter = 0.28;
    const scaleR = 0.35;
    const scaleY = 0.55;
    const grid = [];

    for (let i = 0; i <= N; i++) {
      const u = i / N;
      const { point, tangent } = evalActiveCurve(u);
      const radius = Math.max(0.02, Math.abs(point[0] - axisX) * scaleR);
      const yVal = point[1] * scaleY;
      const dr = (point[0] >= axisX ? 1 : -1) * tangent[0] * scaleR;
      const dy = tangent[1] * scaleY;
      const profileLen = Math.hypot(dr, dy) || 1;
      const nr2D = dy / profileLen;
      const ny2D = -dr / profileLen;

      const row = [];
      for (let j = 0; j <= M; j++) {
        const phi = (j / M) * Math.PI * 2;
        const cp = Math.cos(phi);
        const sp = Math.sin(phi);
        const pos = [
          axisX + radius * cp,
          yVal,
          zCenter + radius * sp,
        ];
        const norm = vec3.normalize([nr2D * cp, ny2D, nr2D * sp]);
        row.push({ pos, norm });
      }
      grid.push(row);
    }

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        const v00 = grid[i][j];
        const v10 = grid[i + 1][j];
        const v11 = grid[i + 1][j + 1];
        const v01 = grid[i][j + 1];
        pushTri(v00.pos, v00.norm, v10.pos, v10.norm, v11.pos, v11.norm);
        pushTri(v00.pos, v00.norm, v11.pos, v11.norm, v01.pos, v01.norm);
      }
    }

    const data = new Float32Array(verts);
    sweptVertexCount = data.length / 6;
    if (data.byteLength > sweptVBOSize) {
      sweptVBO.destroy();
      sweptVBOSize = Math.max(data.byteLength, sweptVBOSize * 2);
      sweptVBO = gpu.device.createBuffer({
        label: 'swept curve mesh vbo',
        size: sweptVBOSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    gpu.device.queue.writeBuffer(sweptVBO, 0, data);
  }

  function appendWireCube(segments, model, colour) {
    const rgb = [colour[0], colour[1], colour[2]];
    const corners = [
      [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
      [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
      [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
      [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ].map((c) => {
      const w = mat4.transformVec4(model, [c[0], c[1], c[2], 1.0]);
      return [w[0], w[1], w[2]];
    });
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [i, j] of edges) {
      segments.push(...corners[i], ...rgb, ...corners[j], ...rgb);
    }
  }

  function update(dt, time) {
    if (camCtrl.viewMode === 'main') {
      mainCam.projection = camera.projection;
    }

    if (camTransition) {
      camTransition.elapsed += dt;
      const rawT = Math.min(1.0, camTransition.elapsed / camTransition.duration);
      cameraQuat = quat.slerp(camTransition.qStart, camTransition.qEnd, rawT);

      camCtrl.distance = (1 - rawT) * camTransition.distStart + rawT * camTransition.distEnd;
      camCtrl.fovDeg = (1 - rawT) * camTransition.fovStart + rawT * camTransition.fovEnd;
      mainCam.fovY = (camCtrl.fovDeg * Math.PI) / 180;
      mainCam.target = [
        (1 - rawT) * camTransition.targetStart[0] + rawT * camTransition.targetEnd[0],
        (1 - rawT) * camTransition.targetStart[1] + rawT * camTransition.targetEnd[1],
        (1 - rawT) * camTransition.targetStart[2] + rawT * camTransition.targetEnd[2],
      ];
      mainCam.eye = vec3.add(mainCam.target, quat.rotateVec3(cameraQuat, [0, 0, camCtrl.distance]));
      applyActiveCamera();

      if (rawT >= 1.0) {
        camCtrl.azimuthDeg = camTransition.azEnd;
        camCtrl.elevationDeg = camTransition.elEnd;
        camTransition = null;
        syncCamerasFromControls();
        for (const ctrl of uiControllers) ctrl.update();
      }
    }

    const T = mat4.translate(state.tx, state.ty, state.tz);
    const R = mulAll(
      mat4.rotateZ(state.rz),
      mat4.rotateY(state.ry),
      mat4.rotateX(state.rx)
    );
    const S = mat4.scale(state.sx, state.sy, state.sz);

    matTRS = mulAll(T, R, S);
    matSRT = mulAll(S, R, T);

    activePartAModel = state.useTRS ? matTRS : matSRT;
    alternatePartAModel = state.useTRS ? matSRT : matTRS;

    secondObjectModel = mulAll(
      mat4.translate(-0.52, -0.42, 0.42),
      mat4.rotateZ(0.12),
      mat4.rotateY(0.45 + 0.15 * Math.sin(time * 0.7)),
      mat4.scale(0.28, 0.18, 0.20)
    );

    const buoyBob = 0.035 * Math.sin(time * 2.1);
    const buoyTilt = 0.18 * Math.cos(time * 1.7);
    thirdObjectModel = mulAll(
      mat4.translate(-0.08, -0.64 + buoyBob, 0.62),
      mat4.rotateZ(buoyTilt),
      mat4.rotateY(time * 0.8),
      mat4.scale(0.13, 0.13, 0.13)
    );

    readoutA.set(
      `T · R · S\n${mat4.format(matTRS, 2)}\n\n` +
      `S · R · T\n${mat4.format(matSRT, 2)}`
    );

    const deg2rad = Math.PI / 180;
    const tAnim = state.animateHierarchy ? time * state.animSpeed : 0;

    const hullWaveY = state.animateHierarchy ? 0.025 * Math.sin(tAnim * 1.6) : 0;
    const hullRoll = (state.hullRollDeg * deg2rad) + (state.animateHierarchy ? 0.08 * Math.sin(tAnim * 1.3) : 0);
    const hullPitchWave = state.animateHierarchy ? 0.035 * Math.cos(tAnim * 1.1) : 0;
    const mastYaw = (state.mastYawDeg * deg2rad) + (state.animateHierarchy ? 0.22 * Math.sin(tAnim * 0.9) : 0);
    const boomShoulder = (state.boomAngleDeg * deg2rad) + (state.animateHierarchy ? 0.32 * Math.sin(tAnim * 1.4) : 0);
    const gaffElbow = (state.gaffAngleDeg * deg2rad) + (state.animateHierarchy ? 0.45 * Math.cos(tAnim * 2.0) : 0);

    boatHull.local = mulAll(
      mat4.translate(0.36, -0.46 + hullWaveY, 0.45),
      mat4.rotateY(-0.18),
      mat4.rotateZ(hullPitchWave),
      mat4.rotateX(hullRoll)
    );

    deckCabin.local = mulAll(
      mat4.translate(-0.12, 0.06, 0.0),
      mat4.rotateY(state.animateHierarchy ? 0.1 * Math.sin(tAnim * 2.2) : 0)
    );

    mainMast.local = mulAll(
      mat4.translate(0.05, 0.05, 0.0),
      mat4.rotateY(mastYaw)
    );

    boomArm.local = mulAll(
      mat4.translate(0.0, 0.34, 0.0),
      mat4.rotateZ(boomShoulder)
    );

    gaffSpar.local = mulAll(
      mat4.translate(0.34, 0.0, 0.0),
      mat4.rotateZ(gaffElbow)
    );

    const flagFlutter = state.animateHierarchy ? 0.55 * Math.sin(tAnim * 4.5) : 0.2;
    pennantFlag.local = mulAll(
      mat4.translate(0.26, 0.0, 0.0),
      mat4.rotateY(flagFlutter)
    );

    boatHull.updateWorld();

    if (state.animateCurveObject) {
      curveRunnerU = (curveRunnerU + dt * state.curveAnimSpeed) % 1.0;
    } else {
      curveRunnerU = state.deCasteljauT;
    }

    if (state.showComputeRibbon) {
      computeRibbon.evaluate(getBezierSegments(), 0.024, 0.014);
    }

    if (state.showSweptMesh) {
      updateSweptMeshVBO();
    }
  }

  function draw(pass, viewProj) {
    meshRenderer.beginFrame();
    lineRenderer.beginFrame();

    const lineSegments = [];

    const activeColour = state.useTRS ? [0.36, 0.42, 0.52, 1.0] : [0.20, 0.68, 0.88, 1.0];
    meshRenderer.drawCube(pass, viewProj, activePartAModel, activeColour, renderer);

    if (state.showGhostOrder) {
      const ghostColour = state.useTRS ? [0.25, 0.78, 1.0, 1.0] : [1.0, 0.55, 0.20, 1.0];
      appendWireCube(lineSegments, alternatePartAModel, ghostColour);
    }

    meshRenderer.drawCube(pass, viewProj, secondObjectModel, [0.85, 0.36, 0.16, 1.0], renderer);
    meshRenderer.drawCube(pass, viewProj, thirdObjectModel, [0.90, 0.22, 0.18, 1.0], renderer);

    for (const item of hierarchyNodes) {
      const geomLocal = mulAll(
        mat4.translate(item.meshOffset[0], item.meshOffset[1], item.meshOffset[2]),
        mat4.scale(item.meshScale[0], item.meshScale[1], item.meshScale[2])
      );
      const geomWorld = mat4.multiply(item.node.world, geomLocal);
      meshRenderer.drawCube(pass, viewProj, geomWorld, item.colour, renderer);

      if (state.debugAxes) {
        lineRenderer.drawAxes(pass, viewProj, item.node.world, item.axesSize, renderer);
      }
    }

    if (state.debugAxes) {
      const partAFrame = mulAll(
        mat4.translate(state.tx, state.ty, state.tz),
        mat4.rotateZ(state.rz),
        mat4.rotateY(state.ry),
        mat4.rotateX(state.rx)
      );
      lineRenderer.drawAxes(pass, viewProj, partAFrame, 0.22, renderer);
    }

    if (state.showComputeRibbon) {
      meshRenderer.drawMesh(
        pass,
        computeRibbon.vbo,
        computeRibbon.vertexCount,
        viewProj,
        mat4.identity(),
        [0.98, 0.62, 0.12, 1.0],
        renderer
      );
    }

    if (state.showSweptMesh && sweptVertexCount > 0) {
      meshRenderer.drawMesh(
        pass,
        sweptVBO,
        sweptVertexCount,
        viewProj,
        mat4.identity(),
        [0.24, 0.68, 0.78, 1.0],
        renderer
      );
    }

    const zLift = 0.012;

    if (state.showControlPolygon) {
      const polyCol = [0.62, 0.66, 0.72];
      for (let i = 0; i < controlPoints.length - 1; i++) {
        const a = controlPoints[i];
        const b = controlPoints[i + 1];
        lineSegments.push(a[0], a[1], zLift, ...polyCol, b[0], b[1], zLift, ...polyCol);
      }

      for (let i = 0; i < controlPoints.length; i++) {
        const p = controlPoints[i];
        const isKnot = (i % 3 === 0);
        const isSelected = (i === selectedPointIndex);
        const size = isSelected ? 0.036 : 0.026;
        const ptColour = isSelected
          ? [1.0, 0.95, 0.15, 1.0]
          : isKnot
            ? [0.15, 0.92, 0.45, 1.0]
            : [0.95, 0.35, 0.70, 1.0];
        const ptModel = mulAll(
          mat4.translate(p[0], p[1], zLift),
          mat4.scale(size, size, size * 0.5)
        );
        meshRenderer.drawCube(pass, viewProj, ptModel, ptColour, renderer);
      }
    }

    const col = state.curveFamily === 'bezier' ? [1.0, 0.78, 0.12] : [0.18, 0.88, 0.95];
    const SAMPLES = 96;
    let prev = evalActiveCurve(0).point;
    for (let s = 1; s <= SAMPLES; s++) {
      const u = s / SAMPLES;
      const curr = evalActiveCurve(u).point;
      lineSegments.push(
        prev[0], prev[1], zLift + 0.003, ...col,
        curr[0], curr[1], zLift + 0.003, ...col
      );
      prev = curr;
    }

    if (tracedStroke.length >= 2) {
      const strokeCol = [0.25, 1.0, 0.45];
      for (let i = 0; i < tracedStroke.length - 1; i++) {
        const a = tracedStroke[i];
        const b = tracedStroke[i + 1];
        lineSegments.push(a[0], a[1], zLift + 0.008, ...strokeCol, b[0], b[1], zLift + 0.008, ...strokeCol);
      }
    }

    if (state.showDeCasteljau) {
      const numSegs = (controlPoints.length - 1) / 3;
      const s = Math.min(state.deCasteljauT * numSegs, numSegs - 1e-6);
      const seg = Math.max(0, Math.floor(s));
      const localT = s - seg;
      const idx = seg * 3;
      const levels = curves.deCasteljau(
        [
          controlPoints[idx],
          controlPoints[idx + 1],
          controlPoints[idx + 2],
          controlPoints[idx + 3],
        ],
        localT
      );
      const level1 = levels[1];
      const level2 = levels[2];
      const ptB = levels[3][0];

      const c1 = [0.20, 0.95, 0.45];
      for (let i = 0; i < 2; i++) {
        const a = level1[i];
        const b = level1[i + 1];
        lineSegments.push(a[0], a[1], zLift + 0.006, ...c1, b[0], b[1], zLift + 0.006, ...c1);
      }

      const c2 = [0.25, 0.72, 1.0];
      lineSegments.push(
        level2[0][0], level2[0][1], zLift + 0.008, ...c2,
        level2[1][0], level2[1][1], zLift + 0.008, ...c2
      );

      const bModel = mulAll(
        mat4.translate(ptB[0], ptB[1], zLift + 0.015),
        mat4.scale(0.03, 0.03, 0.03)
      );
      meshRenderer.drawCube(pass, viewProj, bModel, [1.0, 1.0, 1.0, 1.0], renderer);
    }

    {
      const { point, tangent } = evalActiveCurve(curveRunnerU);
      const T = vec3.normalize(tangent);
      const angleZ = Math.atan2(T[1], T[0]);
      const runnerPos = [point[0], point[1], 0.04];

      const runnerModel = mulAll(
        mat4.translate(runnerPos[0], runnerPos[1], runnerPos[2]),
        mat4.rotateZ(angleZ),
        mat4.scale(0.08, 0.03, 0.03)
      );
      meshRenderer.drawCube(pass, viewProj, runnerModel, [0.95, 0.52, 0.14, 1.0], renderer);
    }

    if (state.showPerspectiveOverlay) {
      const yHorizon = 0.18;
      const vpQuay = [-0.18, yHorizon, zLift];
      const hCol = [1.0, 0.86, 0.22];
      const vCol = [0.25, 0.92, 1.0];

      lineSegments.push(-1.24, yHorizon, zLift, ...hCol, 1.24, yHorizon, zLift, ...hCol);

      const origins = [
        [-1.22, -0.92, zLift],
        [-0.88, -0.95, zLift],
        [-0.48, -0.95, zLift],
        [ 0.78, -0.88, zLift],
      ];
      for (const orig of origins) {
        lineSegments.push(...orig, ...vCol, ...vpQuay, ...vCol);
      }

      const vpMat = mulAll(
        mat4.translate(vpQuay[0], vpQuay[1], vpQuay[2]),
        mat4.scale(0.036, 0.036, 0.015)
      );
      meshRenderer.drawCube(pass, viewProj, vpMat, [1.0, 0.86, 0.22, 1.0], renderer);
    }

    if (camCtrl.viewMode === 'inspection') {
      const aspect = gpu.canvas ? gpu.canvas.width / gpu.canvas.height : 1.5;
      appendCameraFrustum(
        lineSegments,
        {
          eye: mainCam.eye,
          target: mainCam.target,
          up: mainCam.up,
          projection: mainCam.projection,
          fovY: mainCam.fovY,
          aspect,
          near: mainCam.near,
          far: mainCam.far,
          orthoHalfH: mainCam.orthoHeight * 0.5,
        },
        pass,
        viewProj
      );
    }

    if (lineSegments.length > 0) {
      lineRenderer.drawLines(pass, viewProj, new Float32Array(lineSegments), renderer);
    }
  }

  function appendCameraFrustum(segments, camInfo, pass, viewProj) {
    const { eye, target, up, projection, fovY, aspect, near, far, orthoHalfH } = camInfo;

    const f = vec3.normalize(vec3.sub(target, eye));
    let r = vec3.cross(f, up);
    if (vec3.length(r) < 1e-6) {
      const fallbackUp = Math.abs(f[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      r = vec3.cross(f, fallbackUp);
    }
    r = vec3.normalize(r);
    const u = vec3.cross(r, f);

    const camFrame = new Float32Array([
      r[0], r[1], r[2], 0,
      u[0], u[1], u[2], 0,
      -f[0], -f[1], -f[2], 0,
      eye[0], eye[1], eye[2], 1,
    ]);
    const camBody = mulAll(camFrame, mat4.scale(0.10, 0.07, 0.12));
    meshRenderer.drawCube(pass, viewProj, camBody, [0.95, 0.80, 0.18, 1.0], renderer);
    lineRenderer.drawAxes(pass, viewProj, camFrame, 0.22, renderer);

    const visFar = Math.min(far, 4.2);
    let hn, wn, hf, wf;
    if (projection === 'perspective') {
      const tanHalf = Math.tan(fovY * 0.5);
      hn = near * tanHalf;
      wn = hn * aspect;
      hf = visFar * tanHalf;
      wf = hf * aspect;
    } else {
      hn = orthoHalfH;
      wn = orthoHalfH * aspect;
      hf = hn;
      wf = wn;
    }

    const nCenter = vec3.add(eye, vec3.scale(f, near));
    const fCenter = vec3.add(eye, vec3.scale(f, visFar));

    const planeCorners = (center, hw, hh) => [
      vec3.add(center, vec3.add(vec3.scale(r, -hw), vec3.scale(u, -hh))),
      vec3.add(center, vec3.add(vec3.scale(r,  hw), vec3.scale(u, -hh))),
      vec3.add(center, vec3.add(vec3.scale(r,  hw), vec3.scale(u,  hh))),
      vec3.add(center, vec3.add(vec3.scale(r, -hw), vec3.scale(u,  hh))),
    ];

    const nC = planeCorners(nCenter, wn, hn);
    const fC = planeCorners(fCenter, wf, hf);

    const colNear = [0.25, 0.95, 0.55];
    const colFar  = [0.95, 0.55, 0.20];
    const colEdge = [0.95, 0.88, 0.30];
    const colSight = [0.65, 0.75, 1.0];

    for (let i = 0; i < 4; i++) {
      const next = (i + 1) % 4;
      segments.push(...nC[i], ...colNear, ...nC[next], ...colNear);
      segments.push(...fC[i], ...colFar, ...fC[next], ...colFar);
      segments.push(...eye, ...colEdge, ...nC[i], ...colEdge);
      segments.push(...nC[i], ...colEdge, ...fC[i], ...colEdge);
    }

    segments.push(...eye, ...colSight, ...target, ...colSight);
  }

  function pickControlPoint(worldPtOnZ0, maxRadius = 0.075) {
    let bestIdx = -1;
    let bestDist = maxRadius;
    for (let i = 0; i < controlPoints.length; i++) {
      const p = controlPoints[i];
      const d = Math.hypot(p[0] - worldPtOnZ0[0], p[1] - worldPtOnZ0[1]);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function moveControlPoint(idx, newWorldPt) {
    if (idx < 0 || idx >= controlPoints.length) return;
    const clampedX = Math.max(-1.6, Math.min(1.6, newWorldPt[0]));
    const clampedY = Math.max(-1.15, Math.min(1.15, newWorldPt[1]));
    const oldPt = controlPoints[idx];
    const dx = clampedX - oldPt[0];
    const dy = clampedY - oldPt[1];

    controlPoints[idx] = [clampedX, clampedY, 0.0];

    if (idx === 3 || idx === 6) {
      controlPoints[idx - 1] = [
        controlPoints[idx - 1][0] + dx,
        controlPoints[idx - 1][1] + dy,
        0.0,
      ];
      controlPoints[idx + 1] = [
        controlPoints[idx + 1][0] + dx,
        controlPoints[idx + 1][1] + dy,
        0.0,
      ];
    } else if (idx === 2) {
      controlPoints[4] = curves.c1NextControl(controlPoints[2], controlPoints[3]);
    } else if (idx === 4) {
      controlPoints[2] = curves.c1NextControl(controlPoints[4], controlPoints[3]);
    } else if (idx === 5) {
      controlPoints[7] = curves.c1NextControl(controlPoints[5], controlPoints[6]);
    } else if (idx === 7) {
      controlPoints[5] = curves.c1NextControl(controlPoints[7], controlPoints[6]);
    }
  }

  function onPointerDown(e, canvas) {
    const aspect = canvas.width / canvas.height;
    if (e.button === 0 && !e.shift && !e.alt) {
      const hitPt = intersectRayZ0(camera, aspect, e.ndc[0], e.ndc[1]);
      if (state.freehandTraceMode && hitPt) {
        isTracingStroke = true;
        tracedStroke = [hitPt];
        selectedPointIndex = -1;
        return;
      }
      if (hitPt) {
        const picked = pickControlPoint(hitPt);
        if (picked >= 0) {
          isDraggingPoint = true;
          selectedPointIndex = picked;
          return;
        }
      }
    }
    isDraggingPoint = false;
    isTracingStroke = false;
    selectedPointIndex = -1;
  }

  function onPointerDrag(e, canvas) {
    const aspect = canvas.width / canvas.height;

    if (isTracingStroke && state.freehandTraceMode) {
      const hitPt = intersectRayZ0(camera, aspect, e.ndc[0], e.ndc[1]);
      if (hitPt) {
        const last = tracedStroke[tracedStroke.length - 1];
        if (!last || Math.hypot(hitPt[0] - last[0], hitPt[1] - last[1]) > 0.015) {
          tracedStroke.push(hitPt);
        }
      }
      return;
    }

    if (isDraggingPoint && selectedPointIndex >= 0) {
      const hitPt = intersectRayZ0(camera, aspect, e.ndc[0], e.ndc[1]);
      if (hitPt) {
        moveControlPoint(selectedPointIndex, hitPt);
      }
      return;
    }

    camTransition = null;
    if (e.shift || e.button === 1) {
      const activeCam = camCtrl.viewMode === 'inspection' ? inspectionCamera : mainCam;
      const panScale = 0.0025 * (camCtrl.viewMode === 'inspection' ? camCtrl.inspectDistance : camCtrl.distance);
      const f = vec3.normalize(vec3.sub(activeCam.target, activeCam.eye));
      const r = vec3.normalize(vec3.cross(f, activeCam.up));
      const u = vec3.cross(r, f);
      const delta = vec3.add(vec3.scale(r, -e.dx * panScale), vec3.scale(u, e.dy * panScale));
      activeCam.target = vec3.add(activeCam.target, delta);
      syncCamerasFromControls();
    } else {
      if (camCtrl.viewMode === 'inspection') {
        camCtrl.inspectAzimuthDeg -= e.dx * 0.35;
        camCtrl.inspectElevationDeg = Math.max(-85, Math.min(85, camCtrl.inspectElevationDeg + e.dy * 0.35));
      } else {
        camCtrl.azimuthDeg -= e.dx * 0.35;
        camCtrl.elevationDeg = Math.max(-85, Math.min(85, camCtrl.elevationDeg + e.dy * 0.35));
        for (const ctrl of uiControllers) ctrl.update();
      }
      syncCamerasFromControls();
    }
  }

  function onPointerUp() {
    if (isTracingStroke && tracedStroke.length >= 4) {
      const fitted = fitPiecewiseBezierLeastSquares(tracedStroke);
      if (fitted) {
        controlPoints = fitted;
      }
    }
    isTracingStroke = false;
    isDraggingPoint = false;
  }

  function onPointerWheel(e) {
    camTransition = null;
    const zoomFactor = Math.exp(e.delta * 0.001);
    if (camCtrl.viewMode === 'inspection') {
      camCtrl.inspectDistance = Math.max(1.2, Math.min(14.0, camCtrl.inspectDistance * zoomFactor));
    } else {
      camCtrl.distance = Math.max(0.6, Math.min(8.0, camCtrl.distance * zoomFactor));
      mainCam.orthoHeight = Math.max(0.5, Math.min(6.0, mainCam.orthoHeight * zoomFactor));
      for (const ctrl of uiControllers) ctrl.update();
    }
    syncCamerasFromControls();
  }

  function reset() {
    camTransition = null;
    Object.assign(state, defaultState());
    Object.assign(camCtrl, defaultCamControl());
    Object.assign(mainCam, createCamera());
    controlPoints = createInitialControlPoints();
    tracedStroke = [];
    selectedPointIndex = -1;
    curveRunnerU = 0.0;
    viewModeSelect?.set(camCtrl.viewMode);
    familySelect.set(state.curveFamily);
    syncCamerasFromControls();
    for (const ctrl of uiControllers) ctrl.update();
  }

  return {
    state,
    update,
    draw,
    reset,
    setupViewFolder,
    onPointerDown,
    onPointerDrag,
    onPointerUp,
    onPointerWheel,
    getCurveJSON,
  };
}
