// Bake the MIT WebXR hand into identical meshes for Three.js and MuJoCo vision.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import * as THREE from '../frontend/node_modules/three/build/three.module.js';
import { GLTFLoader } from '../frontend/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
const root = fileURLToPath(new URL('../frontend/public/models/hand/', import.meta.url));
const source = fs.readFileSync(root + 'source-right.glb');
const output = root + 'facing-v4/';
fs.mkdirSync(output, {recursive: true});
const poses = {};
for (const gesture of ['palm', 'fist', 'point']) {
  const { scene } = await new GLTFLoader().parseAsync(source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength), '');
  scene.updateMatrixWorld(true);
  const mesh = scene.getObjectByProperty('type', 'SkinnedMesh');
  const chains = [
    ['thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip'],
    ...['index', 'middle', 'ring', 'pinky'].map(f => ['metacarpal', 'phalanx-proximal', 'phalanx-intermediate', 'phalanx-distal', 'tip'].map(j => `${f}-finger-${j}`)),
  ];
  for (const chain of chains) {
    const bones = chain.map(name => scene.getObjectByName(name));
    for (let i = 1; i < bones.length; i++) bones[i - 1].attach(bones[i]);
    if (gesture === 'palm' || (gesture === 'point' && chain[0].startsWith('index'))) continue;
    for (let i = 1; i < bones.length - 1; i++) {
      const bone = bones[i];
      const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(bone.getWorldQuaternion(new THREE.Quaternion()).invert());
      bone.rotateOnAxis(axis, chain[0].startsWith('thumb') ? -.35 : i === 1 ? -1.25 : -1.35);
      scene.updateMatrixWorld(true);
    }
  }
  scene.updateMatrixWorld(true);
  mesh.skeleton.update();
  const vertices = [];
  const v = new THREE.Vector3();
  // All poses use the same scale and origin: 180 mm open-hand height.
  for (let i = 0; i < mesh.geometry.attributes.position.count; i++) {
    mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
    vertices.push((v.x - .035) * 1000, (v.z - .005) * 1000, (-v.y + .057) * 1000);
  }
  const indices = [...mesh.geometry.index.array];
  poses[gesture] = {vertices, indices};
}
// A person faces the fly: human-left is on fly-right (-Y).
// Mirror the source right hand and reverse its triangle winding for human-left.
// Face each palm inward and downward toward the fly, around a shared palm pivot.
const cues = {palm: ['palm', 'palm'], fist: ['fist', 'fist'],
  point: ['point', 'palm'], point_right: ['palm', 'point'], point_both: ['point', 'point']};
for (const [gesture, articulations] of Object.entries(cues)) {
  const vertices = [], indices = [];
  articulations.forEach((pose, side) => {
    const hand = poses[pose], base = vertices.length / 3;
    for (let i = 0; i < hand.vertices.length; i += 3) {
      const [x, y, z] = hand.vertices.slice(i, i + 3);
      const sign = side === 0 ? -1 : 1;
      const yaw = sign * (Math.atan2(100, 110) + 13 * Math.PI / 180);
      const pitch = -50 * Math.PI / 180;
      const lateral = side === 0 ? -y : y;
      const forward = Math.cos(pitch) * (x + 15) + Math.sin(pitch) * (z - 85);
      vertices.push(
        Math.cos(yaw) * forward - Math.sin(yaw) * lateral,
        Math.sin(yaw) * forward + Math.cos(yaw) * lateral + sign * 100,
        -Math.sin(pitch) * (x + 15) + Math.cos(pitch) * (z - 85),
      );
    }
    for (let i = 0; i < hand.indices.length; i += 3) {
      const [a, b, c] = hand.indices.slice(i, i + 3);
      indices.push(base + a, base + (side === 0 ? c : b), base + (side === 0 ? b : c));
    }
  });
  fs.writeFileSync(output + `${gesture}.json`, JSON.stringify({vertices, indices}));
  let obj = '# Paired posed WebXR hands; MIT, Copyright (c) 2019 Amazon\n';
  for (let i = 0; i < vertices.length; i += 3) obj += `v ${vertices.slice(i, i + 3).join(' ')}\n`;
  for (let i = 0; i < indices.length; i += 3) obj += `f ${indices.slice(i, i + 3).map(x => x + 1).join(' ')}\n`;
  fs.writeFileSync(output + `${gesture}.obj`, obj);
}
fs.writeFileSync(output + 'provenance.json', JSON.stringify({
  source: 'https://github.com/immersive-web/webxr-input-profiles/tree/main/packages/assets/profiles/generic-hand',
  source_sha256: crypto.createHash('sha256').update(source).digest('hex'),
  license: 'MIT; Copyright (c) 2019 Amazon',
  modifications: 'Reparented finger bones, posed joints, baked mesh to millimetres; mirrored human-left hand with reversed winding at -100 mm Y, human-right at +100 mm Y; palms centered at (-15,0,85), yawed inward by atan2(100,110)+13 degrees, tilted -50 degrees. Human hand labels map to the same-named fly front legs.',
  generator: 'scripts/build_hand_gestures.mjs',
}, null, 2));
