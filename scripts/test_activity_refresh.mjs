import test from 'node:test';
import assert from 'node:assert/strict';
import { activityRefresh } from '../frontend/src/lib/activityRefresh.ts';
const flush = () => new Promise(resolve => setImmediate(resolve));

test('one request at a time, latest frame wins, unchanged state stays idle', async () => {
  const calls = [], finish = [];
  const updater = activityRefresh(stamp => {
    calls.push(stamp);
    return new Promise(resolve => finish.push(resolve));
  }, () => true);
  updater.request('1:0');
  updater.request('1:1');
  updater.request('1:2');
  assert.deepEqual(calls, ['1:0']);
  finish.shift()(); await flush();
  assert.deepEqual(calls, ['1:0', '1:2']);
  finish.shift()(); await flush();
  updater.request('1:2'); await flush();
  assert.equal(calls.length, 2);
  updater.dispose();
});

test('hidden views defer work and resume from the latest stamp', async () => {
  let visible = false;
  const calls = [];
  const updater = activityRefresh(async stamp => { calls.push(stamp); }, () => visible);
  updater.request('1:0'); updater.request('1:1');
  assert.equal(calls.length, 0);
  visible = true; updater.request('1:1'); await flush();
  assert.deepEqual(calls, ['1:1']);
  updater.dispose();
});

test('disposing while a request runs prevents queued follow-up work', async () => {
  let finish;
  const calls = [];
  const updater = activityRefresh(stamp => {
    calls.push(stamp); return new Promise(resolve => { finish = resolve; });
  }, () => true);
  updater.request('1:0'); updater.request('2:0'); updater.dispose();
  finish(); await flush();
  assert.deepEqual(calls, ['1:0']);
});

test('failed fetch backs off and retries the newest frame', async () => {
  const calls = [];
  let success;
  const done = new Promise(resolve => { success = resolve; });
  const updater = activityRefresh(async stamp => {
    calls.push(stamp);
    if (calls.length === 1) throw new Error('Temporary outage');
    success();
  }, () => true, 10);
  updater.request('1:0'); await flush(); updater.request('1:3');
  assert.deepEqual(calls, ['1:0']);
  await done;
  assert.deepEqual(calls, ['1:0', '1:3']);
  updater.dispose();
});
