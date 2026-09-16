from dataclasses import asdict

import numpy as np
import torch

from flylab.retina import BALANCED_MODEL, optical_geometry
from flylab.policy_model import SensorActionTransformer, SensorSpec
from flylab.policy_simulation import PolicySimulation
from flylab.body import BodyParameters, FlyBody
from flylab.senses import SensorySettings


def test_equal_groups_cover_distinct_mirrored_directions():
    left, right = optical_geometry(BALANCED_MODEL)
    assert left[0].shape == right[0].shape == (1024, 3)
    assert left[1].shape == right[1].shape == (7168, 3)
    assert len(np.unique(left[0], axis=0)) == 1024
    np.testing.assert_allclose(np.linalg.norm(left[0], axis=1), 1)
    np.testing.assert_allclose(right[0], left[0] * [1, -1, 1])
    assert not np.allclose(left[0], right[0])
    model = SensorActionTransformer(width=32, layers=1)
    for tokenizer in model.tokenizers.values():
        pool = tokenizer.pool.numpy()
        assert pool.shape == (64, 1024)
        assert ((pool > 0).sum(1) == 16).all()
        assert ((pool > 0).sum(0) == 1).all()
        np.testing.assert_allclose(pool.sum(1), 1)


def test_new_and_legacy_schemas_use_their_own_optics():
    body = FlyBody()
    names = [body.model.actuator(i).name for i in range(84)]
    settings = {'body': asdict(BodyParameters()), 'senses': asdict(SensorySettings(vision_enabled=True))}
    for sensors, counts in [(None, [1024, 1024]),
                            ([SensorSpec('left_eye', 857, tokens=64, geometry='left'),
                              SensorSpec('right_eye', 852, tokens=64, geometry='right')], [857, 852])]:
        model = SensorActionTransformer(sensors=sensors, action_names=names, width=32, layers=1)
        sim = PolicySimulation(model, settings)
        sim.advance()
        assert [eye['count'] for eye in sim.snapshot()['senses']['vision']['eyes']] == counts
        restored = SensorActionTransformer.from_schema(model.schema())
        restored.load_state_dict(model.state_dict())
        inputs = {spec.name: torch.zeros(1, 8, spec.samples, spec.channels) for spec in model.sensors}
        torch.testing.assert_close(model(inputs), restored(inputs))
    assert [len(axes) for axes, _ in optical_geometry()] == [857, 852]


def test_resampling_follows_measured_coverage_and_density():
    from flylab.retina import eye_template
    measured = np.asarray(eye_template()['right'])
    measured /= np.linalg.norm(measured, axis=1)[:, None]
    sampled = optical_geometry(BALANCED_MODEL)[1][0]
    # No new broad angular regions: every designed ray remains close to the
    # measured surface, unlike the previous rectangular angular field.
    nearest = np.rad2deg(np.arccos(np.clip((sampled @ measured.T).max(1), -1, 1)))
    assert nearest.max() < 5
    def density(axes):
        az = np.arctan2(-axes[:, 1], axes[:, 0])
        el = np.arcsin(axes[:, 2])
        hist = np.histogram2d(az, el, bins=(8, 6),
            range=((-np.pi/2, 3*np.pi/2), (-np.pi/2, np.pi/2)))[0]
        return hist / hist.sum()
    assert np.abs(density(measured) - density(sampled)).sum() < .1


def test_old_grid_schema_keeps_its_original_directions():
    from flylab.retina import GRID_MODEL
    sensors = [SensorSpec('left_eye', 1024, tokens=64, geometry='balanced-left'),
               SensorSpec('right_eye', 1024, tokens=64, geometry='balanced-right')]
    model = SensorActionTransformer(sensors=sensors, width=32, layers=1)
    for tokenizer, (axes, _) in zip(model.tokenizers.values(), optical_geometry(GRID_MODEL)):
        np.testing.assert_allclose(tokenizer.axes.numpy(), axes, atol=1e-7)
    assert not np.allclose(optical_geometry(GRID_MODEL)[0][0], optical_geometry(BALANCED_MODEL)[0][0])
