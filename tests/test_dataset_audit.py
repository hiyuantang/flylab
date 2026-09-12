"""Audit helpers must preserve unknown IDs, direction, and integer weights."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('audit_malecns', Path(__file__).parents[1] / 'scripts/audit_malecns.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_unknown_segment_does_not_alias_an_annotated_neuron():
    np.testing.assert_array_equal(audit.lookup(np.array([10, 30]), np.array([0, 10, 20, 30, 99])), [-1, 0, -1, 1, -1])
    np.testing.assert_array_equal(audit.lookup(np.array([], dtype=int), np.array([4])), [-1])


def test_traversal_direction_includes_feedback_and_missing_values():
    np.testing.assert_array_equal(audit.layer_codes(np.array([1., 3., 5., np.nan, 2.]), np.array([4., 3., 2., 1., np.nan])), [0, 1, 2, 3, 3])


def test_synapse_tally_preserves_integer_precision():
    value = 2**53 + 1
    result = audit.tally(np.array([0, 0, 2]), np.array([value, 3, 7], dtype=np.int64), 4)
    np.testing.assert_array_equal(result, [[2, value + 3], [0, 0], [1, 7], [0, 0]])
