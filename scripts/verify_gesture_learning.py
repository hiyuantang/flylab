"""Isolated, reproducible training audit; never changes live or user weights.

Use --settings-from with a completed checkpoint to reproduce its body/senses.
Its adapter is not inherited: the audit starts from the original connectome.
"""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from flylab.gesture_batch import BatchGestureTrainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings-from',type=Path,required=True)
    parser.add_argument('--graph',type=Path,default=Path('data/full'))
    parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    saved=torch.load(args.settings_from,map_location='cpu',weights_only=True)
    with tempfile.TemporaryDirectory(prefix='flylab-learning-audit-') as directory:
        trainer=BatchGestureTrainer(Path(directory),args.graph)
        publish=trainer.publish
        def record(**values):
            publish(**values)
            if 'history' in values:
                print(json.dumps(values['history'][-1]),flush=True)
            elif 'phase' in values:
                print(values['phase'],flush=True)
        trainer.publish=record
        trainer.start(saved['settings'],iterations=args.iterations,horizon=args.steps,
                      batch_size=args.batch_size,rank=saved['rank'],seed=saved['seed'],
                      learning_rate=saved['learning_rate'],proportions={'point':1,'point_right':1})
        trainer.thread.join()
        result=trainer.snapshot(False)
        result.pop('versions',None)
        result['configuration_source']=str(args.settings_from)
        result['starts_from']='original connectome'
        result['checkpoint']='temporary audit only; not installed'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({key:result.get(key) for key in ('error','wall_seconds','validation_initial_loss','validation_loss','adapter','early_stopped')}),flush=True)
        if result.get('error'):
            raise RuntimeError(result['error'])


if __name__=='__main__':
    main()
