import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from flylab.env import FlyEnv


def test_gym_contract_seed_and_time_limit():
    env = FlyEnv(max_steps=20)
    check_env(env, skip_render_check=True)
    a, info_a = env.reset(seed=42)
    env.step(np.full(6, .4, dtype=np.float32))
    b, info_b = env.reset(seed=42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(info_a['target_mm'], info_b['target_mm'])
    env.max_steps = 2
    env.reset(seed=1)
    action = np.full(6, .4, dtype=np.float32)
    env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward) and not terminated and truncated
    assert reward == pytest.approx(sum(info['reward_terms'].values()))
    assert info['simulation_time'] == pytest.approx(.04)
    with pytest.raises(RuntimeError):
        env.step(action)


def test_direct_muscle_actions_and_invalid_input():
    env = FlyEnv(action_mode='muscle')
    assert env.action_space.shape == (84,)
    env.reset(seed=2)
    with pytest.raises(ValueError):
        env.step(np.full(84, np.nan, dtype=np.float32))
    assert env.steps == 0
    env.step(np.full(84, .15, dtype=np.float32))
    assert np.all(env.body.data.act > .1)
    assert np.allclose(env.body.data.ctrl, .15)
