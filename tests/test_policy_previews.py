"""Recorded evaluation samples stay separate and survive trainer restarts."""
import pytest

from flylab.policy_training import PolicyTrainer


def test_preview_roundtrip_across_epochs_and_restart(tmp_path):
    trainer = PolicyTrainer(tmp_path)
    name = 'policy-123.pt'
    (tmp_path / name).touch()
    for epoch, sample in [(10, 1), (10, 2), (20, 1)]:
        trainer.save_preview(name, epoch, sample, {'epoch': epoch, 'sample': sample, 'body': {'pose': [epoch, sample]}})
    restored = PolicyTrainer(tmp_path)
    assert restored.load_preview(name, 10, 2)['body']['pose'] == [10, 2]
    assert restored.load_preview(name, 20, 1)['body']['pose'] == [20, 1]
    with pytest.raises(FileNotFoundError):
        restored.load_preview(name, 20, 2)
    with pytest.raises(ValueError):
        restored.load_preview('../policy-123.pt', 10, 1)
    (tmp_path / name).unlink()
    with pytest.raises(ValueError):
        restored.load_preview(name, 10, 1)
