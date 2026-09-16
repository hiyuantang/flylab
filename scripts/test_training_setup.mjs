import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultTrainingSetup, setupFromVersion, sameTrainingSetup, syncTrainingDraft } from '../frontend/src/lib/trainingSetup.ts';
import { trainingRequestGate } from '../frontend/src/lib/trainingRequests.ts';

const version = {
  id:'policy-parent.pt', chart_id:'chart', chart_name:'Pointing', parent:null,
  training_recipe:'action-chunk-bc-v1', iterations:17, batch_size:8,
  learning_rate:.002, horizon:45, seed:9, rank:0,
  early_stopping:false, early_stopping_patience:6, demonstrations_per_gesture:12,
  validation_episodes:3, rollout_episodes:2, evaluation_interval:5,
  proportions:{palm:0,fist:0,point:50,point_right:50,point_both:0},
};
const draft = (setup=defaultTrainingSetup('transformer')) => ({setup,observedRun:null,completedRun:null,runSetup:null});
const running = (setup) => ({run_id:'run-1',running:true,phase:'Training action chunks',training_setup:setup,versions:[version]});

test('saved selection restores all training inputs; vanilla resets them', () => {
  const selected=setupFromVersion('transformer',version);
  for(const key of ['iterations','batch_size','learning_rate','seed','horizon','early_stopping','early_stopping_patience','demonstrations_per_gesture','validation_episodes','rollout_episodes','evaluation_interval','proportions'])
    assert.deepEqual(selected[key],version[key]);
  assert.equal(selected.checkpoint,version.id);
  assert.equal(selected.version_name,'');
  assert.equal(selected.rank,2);
  assert.equal(setupFromVersion('transformer').iterations,60);
  assert.equal(setupFromVersion('transformer').checkpoint,'');
  const legacy=setupFromVersion('transformer',{...version,training_recipe:'old',early_stopping:undefined});
  assert.equal(legacy.learning_rate,.0003);
  assert.equal(legacy.early_stopping,true);
  assert.equal(setupFromVersion('connectome',{...version,rank:5}).rank,5);
});

test('stop then change starting weights cannot resume the old run', () => {
  const original={...defaultTrainingSetup('transformer'), iterations:73, early_stopping:false};
  const active=syncTrainingDraft(draft(),running(original),'transformer');
  const paused={...running(original),running:false,resumable:true,checkpoint:'policy-paused.pt'};
  const stopped=syncTrainingDraft(active,paused,'transformer');
  assert.equal(stopped.setup.checkpoint,'');
  assert.equal(sameTrainingSetup(stopped.setup,stopped.runSetup),true);
  const changed={...stopped,setup:setupFromVersion('transformer',version)};
  const polled=syncTrainingDraft(changed,paused,'transformer');
  assert.equal(polled,changed);
  assert.equal(sameTrainingSetup(polled.setup,polled.runSetup),false);
  assert.equal(polled.setup.iterations,17);
  const restarted=syncTrainingDraft(changed,{...running(changed.setup),run_id:'run-2'},'transformer');
  assert.equal(restarted.setup.checkpoint,version.id);
  assert.equal(sameTrainingSetup(restarted.setup,restarted.runSetup),true);
});

test('completed run advances only its parent, once; polling cannot overwrite a later choice', () => {
  const setup={...setupFromVersion('transformer',version),iterations:21,version_name:'Experiment B',resume_from:'best'};
  const active=syncTrainingDraft(draft(setup),running(setup),'transformer');
  const result={...version,id:'policy-result.pt'};
  const done={...running(setup),running:false,phase:'Training complete · gesture checks not passed',checkpoint:result.id,versions:[version,result]};
  const completed=syncTrainingDraft(active,done,'transformer');
  assert.deepEqual(completed.setup,{...setup,checkpoint:result.id});
  const edited={...completed,setup:setupFromVersion('transformer',version)};
  assert.equal(syncTrainingDraft(edited,done,'transformer'),edited);
  assert.equal(syncTrainingDraft(completed,done,'transformer'),completed);
});

test('running checkpoint writes, pauses and failures never promote starting weights', () => {
  const setup=defaultTrainingSetup('transformer');
  const active=syncTrainingDraft(draft(setup),running(setup),'transformer');
  for(const state of [
    {...running(setup),checkpoint:version.id},
    {...running(setup),running:false,resumable:true,checkpoint:version.id},
    {...running(setup),running:false,error:'failed',phase:'Training failed',checkpoint:version.id},
  ]) assert.equal(syncTrainingDraft(active,state,'transformer').setup.checkpoint,'');
});

test('reopening an active run displays the server run setup', () => {
  const setup=setupFromVersion('transformer',version);
  assert.deepEqual(syncTrainingDraft(draft(),running(setup),'transformer').setup,setup);
});

test('old polls and actions are rejected after actions and controller changes', () => {
  const gate=trainingRequestGate();
  const beforeStop=gate.begin();
  const stop=gate.begin();
  assert.equal(gate.accepts(beforeStop),false);
  assert.equal(gate.accepts(stop),true);
  gate.invalidate();
  assert.equal(gate.accepts(stop),false);
  const otherController=gate.begin();
  assert.equal(gate.accepts(otherController),true);
});


test('connectome plateau and unchanged-best completion still advance to the saved child', () => {
  const setup=defaultTrainingSetup('connectome');
  for(const phase of ['Training complete; best validated weights saved',
    'Stopped improving; best validated weights saved', 'Complete; starting weights remained best']) {
    const active=syncTrainingDraft(draft(setup),running(setup),'connectome');
    const done={...running(setup),running:false,phase,checkpoint:'gesture-new.pt',versions:[{...version,id:'gesture-new.pt'}]};
    assert.deepEqual(syncTrainingDraft(active,done,'connectome').setup,{...setup,checkpoint:'gesture-new.pt'});
  }
});
