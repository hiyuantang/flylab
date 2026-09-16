import test from 'node:test';
import assert from 'node:assert/strict';
import { PosePlayback } from '../frontend/src/lib/motion.ts';

function frame(time, x, angle = 0, running = true, episode = 1) {
  const q = [Math.cos(angle/2), 0, 0, Math.sin(angle/2)];
  return {time, episode, running, scene:{id:'lab'}, model:{controller:'connectome'},
    timing:{muscle_command_hz:60}, body:{feet:[], bodies:[
      {name:'c_thorax', parent:null, position:[x,0,0], quaternion:q},
      {name:'leg', parent:'c_thorax', position:[x+Math.cos(angle), Math.sin(angle),0], quaternion:q}
    ]}};
}

test('interpolation keeps joint anchors attached to the rotating parent', () => {
  const motion = new PosePlayback();
  motion.push(frame(0,0),0);
  motion.push(frame(.1,2,Math.PI/2),1);
  motion.renderAt(.05);
  const root=motion.get('c_thorax'), leg=motion.get('leg');
  assert.ok(Math.abs(root.position.x-1)<1e-10);
  assert.ok(Math.abs(root.position.distanceTo(leg.position)-1)<1e-10);
  assert.ok(Math.abs(leg.position.x-1-Math.SQRT1_2)<1e-10);
  assert.ok(Math.abs(leg.position.y-Math.SQRT1_2)<1e-10);
});

test('many smooth display frames occur between sparse physical snapshots', () => {
  const motion = new PosePlayback();
  motion.push(frame(0,0),0);
  motion.push(frame(.05,1),.5);
  motion.push(frame(.1,2),1);
  const xs=[];
  for(let i=0;i<30;i++) {motion.advance(1/60); xs.push(motion.get('c_thorax').position.x);}
  assert.ok(new Set(xs).size>20);
  assert.ok(xs.every((x,i)=>i===0 || x>=xs[i-1]));
  assert.ok(xs.at(-1)<2);
  for(let i=0;i<1000;i++) motion.advance(1/60);
  assert.ok(motion.time<=motion.latestTime);
  assert.ok(motion.get('c_thorax').position.x<=2);
});

test('paused steps reach the exact endpoint and restores clear old motion', () => {
  const motion = new PosePlayback();
  motion.push(frame(0,0,0,false),0);
  motion.push(frame(.1,2,0,false),1);
  motion.advance(.05);
  assert.ok(motion.get('c_thorax').position.x>0 && motion.get('c_thorax').position.x<2);
  for(let i=0;i<20;i++) motion.advance(1/60);
  assert.equal(motion.get('c_thorax').position.x,2);
  motion.push(frame(0,10,0,false,2),2);
  assert.equal(motion.get('c_thorax').position.x,10);
  assert.equal(motion.bufferedFrames,1);
});

test('duplicate packets do not alter cadence and retained memory is bounded', () => {
  const motion = new PosePlayback();
  for(let i=0;i<300;i++) {
    motion.push(frame(i/60,i/60),i/10);
    motion.push(frame(i/60,i/60),i/10+.01);
  }
  assert.ok(motion.bufferedFrames<=120);
  motion.advance(1/60);
  assert.ok(Number.isFinite(motion.get('c_thorax').position.x));
});
