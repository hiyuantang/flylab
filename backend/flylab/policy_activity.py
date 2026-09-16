"""Read-only inference telemetry, separate from checkpointed policy dynamics.

Dots represent tokens or feature channels, not biological neurons. Reduction is
explicit; schematic connectivity must never be presented as attention weights.
"""
import torch


class PolicyActivity:
    def __init__(self, model):
        self.groups = []
        self.pending = {}
        self.revision = 0
        self.cached_revision = -1
        self.cached = None
        self.handles = []
        width = model.config['width']
        token_count = sum(s.tokens for s in model.sensors)

        def register(module, key, label, stage, count, unit, reduction, labels=None):
            self.groups.append(dict(id=key, label=label, stage=stage, count=count,
                                    unit=unit, reduction=reduction, labels=labels))

            def capture(_module, _inputs, output):
                if model.training or torch.is_grad_enabled():
                    return
                value = output.detach()[0].float()
                if reduction == 'token_rms':
                    value = value.square().mean(-1).sqrt()
                elif reduction == 'feature_rms':
                    value = value.square().mean(0).sqrt()
                else:
                    value = value[0].sigmoid()
                self.pending[key] = value
            self.handles.append(module.register_forward_hook(capture))

        for spec in model.sensors:
            register(model.tokenizers[spec.name], spec.name, spec.name.replace('_', ' ').title(),
                     0, spec.tokens, f'RMS across {width} features', 'token_rms')
        for i, layer in enumerate(model.encoder.layers):
            last = i == len(model.encoder.layers) - 1
            register(model.encoder if last else layer, f'encoder_{i + 1}',
                     f'Encoder {i + 1}' + (' + norm' if last else ''), i + 1, width,
                     f'RMS across {token_count} tokens', 'feature_rms')
        offset = len(model.encoder.layers) + 1
        for i, layer in enumerate(model.decoder.layers):
            last = i == len(model.decoder.layers) - 1
            register(model.decoder if last else layer, f'decoder_{i + 1}',
                     f'Decoder {i + 1}' + (' + norm' if last else ''), offset + i, width,
                     f'RMS across {model.chunk} command queries', 'feature_rms')
        register(model.output, 'muscles', 'Muscle commands', offset + len(model.decoder.layers),
                 len(model.action_names), 'Excitation [0, 1] · next command', 'excitation', model.action_names)
        self.handles.append(model.register_forward_hook(self._complete))

    def _complete(self, model, _inputs, _output):
        if not model.training and not torch.is_grad_enabled():
            self.revision += 1

    def snapshot(self, step):
        # Reset creates a fresh trial: never show the previous trial as live data.
        if step == 0 or self.revision == 0:
            return dict(groups=self.groups, values={}, step=step, measured=False)
        if self.cached_revision != self.revision:
            self.cached = {key: value.cpu().tolist() for key, value in self.pending.items()}
            self.cached_revision = self.revision
        return dict(groups=self.groups, values=self.cached, step=step, measured=True)


def policy_activity(sim):
    if not hasattr(sim, 'policy'):
        return None
    if not hasattr(sim, '_policy_activity'):
        sim._policy_activity = PolicyActivity(sim.policy)
    return sim._policy_activity.snapshot(sim.steps)
