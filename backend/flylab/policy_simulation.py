"""Independent visual policy controller using the existing fly body and optics."""
from dataclasses import asdict
import copy
import numpy as np
import torch
from .simulation import Simulation
from .body import FlyBody, BodyParameters
from .senses import SensorSuite, SensorySettings
from .policy_model import eye_observation, observation_window, policy_vision_model


class PolicySimulation(Simulation):
    policy_kind = 'transformer'

    def __init__(self, policy, settings):
        settings = copy.deepcopy(settings)
        settings['senses']['vision_model'] = policy_vision_model(policy.schema())
        self.policy = policy.eval()
        super().__init__()
        # Base snapshot scaffolding uses the non-neural physical category. The
        # overridden advance never executes its posture controller.
        self.controller = 'posture'
        self.body = FlyBody(BodyParameters(**settings['body']), 'lab')
        self.sensors = SensorSuite(SensorySettings(**settings['senses']))
        self.policy_settings = settings
        self.odor, self.intensity = 'none', 0.
        self.reset()
        names = [self.body.model.actuator(i).name for i in range(84)]
        if names != self.policy.action_names:
            raise ValueError('Policy action schema differs from body actuators')

    def configure(self, *args, **kwargs):
        raise ValueError('Load another model to change controller or recorded body settings')

    def configure_senses(self, settings):
        if settings.vision_model != policy_vision_model(self.policy.schema()):
            raise ValueError('This policy requires compound-eye observations')
        super().configure_senses(settings)

    def reset(self):
        super().reset()
        self.visual_history = []

    def _advance_physical(self):
        self.visual_history.append(eye_observation(self.sensory_frame()))
        self.visual_history = self.visual_history[-self.policy.history:]
        observations = observation_window(self.visual_history, self.policy.history)
        device = next(self.policy.parameters()).device
        tensors = {k: torch.tensor(v[None], device=device) for k, v in observations.items()}
        with torch.inference_mode():
            commands = self.policy(tensors)[0, 0].float().cpu().numpy()
        action = np.zeros(self.body.model.nu)
        action[:84] = commands
        self.body.step_muscles(action, dt=.02)
        self.steps += 1
        self.time = self.steps * .02
        self.history.append({'time': self.time, 'neural': 0., 'muscle': float(self.body.data.act.mean()),
                             'speed': float(np.linalg.norm(self.body.data.qvel[:2]))})
        self.history = self.history[-300:]

    def snapshot(self):
        result = super().snapshot()
        result['regions'] = []
        result['body_parameters'] = asdict(self.body.parameters)
        result['model'].update(name='Fly Vision–Action Transformer', controller='transformer', ready=True,
            parameters=sum(p.numel() for p in self.policy.parameters()),
            sensor_schema=self.policy.schema()['sensors'],
            device=str(next(self.policy.parameters()).device), precision='torch.float32',
            assumptions='Learned visual engineering policy; 84 leg excitations, remaining actuators zero')
        return result
