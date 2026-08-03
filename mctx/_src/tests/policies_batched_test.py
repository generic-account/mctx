# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `policies_batched.py`."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import policies_batched
from mctx._src import qtransforms
from mctx._src import seq_halving
from mctx._src import tree as tree_lib
import numpy as np

jax.config.update("jax_threefry_partitionable", False)

ROOT = tree_lib.Tree.ROOT_INDEX

# value_scale=0 removes the backed-up Q contribution from root scores, so
# `gumbel_muzero_policy`'s per-simulation rescoring cannot reorder its picks.
# Under this transform the batched policy must reproduce mctx exactly.
drift_free = functools.partial(
    qtransforms.qtransform_completed_by_mix_value, value_scale=0.0)


def _make_smooth_recurrent_fn(num_actions, embedding_size):
  """A deterministic recurrent_fn with non-degenerate values and rewards."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    nxt = embedding + jnp.sin(action.astype(jnp.float32))[:, None]
    total = jnp.sum(nxt, axis=-1)
    return mctx.RecurrentFnOutput(
        reward=jnp.tanh(total),
        discount=jnp.ones_like(total),
        prior_logits=jnp.stack([total] * num_actions, axis=-1),
        value=jnp.tanh(jnp.mean(nxt, axis=-1)),
    ), nxt

  del embedding_size
  return recurrent_fn


def _make_root(batch_size, num_actions, embedding_size, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  return mctx.RootFnOutput(
      prior_logits=jax.random.normal(keys[0], (batch_size, num_actions)),
      value=jax.random.normal(keys[1], (batch_size,)),
      embedding=jax.random.normal(keys[2], (batch_size, embedding_size)),
  )


def _policy_kwargs(batch_size=4, num_actions=13, embedding_size=5,
                   num_simulations=32, max_num_considered_actions=8,
                   invalid_actions=None, seed=0):
  return {
      "params": (),
      "rng_key": jax.random.PRNGKey(seed + 17),
      "root": _make_root(batch_size, num_actions, embedding_size, seed=seed),
      "recurrent_fn": _make_smooth_recurrent_fn(num_actions, embedding_size),
      "num_simulations": num_simulations,
      "max_num_considered_actions": max_num_considered_actions,
      "invalid_actions": invalid_actions,
  }


class PoliciesBatchedTest(parameterized.TestCase):

  def test_rounds_match_the_visit_schedule(self):
    """The round encoding must reproduce mctx's own schedule exactly."""
    for num_simulations, max_considered in [(32, 16), (16, 16), (8, 4),
                                            (64, 16), (128, 32), (200, 32)]:
      rounds = policies_batched._rounds_from_schedule(  # pylint: disable=protected-access
          max_considered, num_simulations)
      flat = []
      for considered_visit, width in rounds:
        flat.extend([considered_visit] * width)
      expected = list(
          seq_halving.get_sequence_of_considered_visits(
              max_considered, num_simulations))
      self.assertEqual(flat, expected)
      self.assertLess(len(rounds), num_simulations)

  @parameterized.parameters((32, 16), (16, 16), (8, 4), (64, 16), (128, 32))
  def test_matches_mctx_with_drift_free_qtransform(self, num_simulations,
                                                   max_considered):
    """Bitwise-equal visit counts and actions vs `gumbel_muzero_policy`."""
    kwargs = _policy_kwargs(num_simulations=num_simulations,
                            max_num_considered_actions=max_considered)
    expected = mctx.gumbel_muzero_policy(qtransform=drift_free, **kwargs)
    got = mctx.batched_gumbel_muzero_policy(qtransform=drift_free, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(expected.search_tree.summary().visit_counts),
        np.asarray(got.search_tree.summary().visit_counts))
    np.testing.assert_array_equal(np.asarray(expected.action),
                                  np.asarray(got.action))

  @parameterized.parameters((32, 16), (8, 4), (128, 32))
  def test_search_output_is_valid(self, num_simulations, max_considered):
    """Invariants for any schedule, including off-schedule rows."""
    num_actions = 13
    # Make some rows have fewer valid actions than `max_considered`, which
    # exercises the documented static-width fallback.
    invalid = jnp.zeros((4, num_actions), dtype=bool)
    invalid = invalid.at[0, 5:].set(True)
    invalid = invalid.at[1, 3:].set(True)
    kwargs = _policy_kwargs(num_actions=num_actions,
                            num_simulations=num_simulations,
                            max_num_considered_actions=max_considered,
                            invalid_actions=invalid)
    out = mctx.batched_gumbel_muzero_policy(**kwargs)
    visits = out.search_tree.summary().visit_counts
    np.testing.assert_array_equal(
        np.asarray(jnp.sum(visits, axis=-1)),
        np.full((4,), num_simulations, dtype=np.asarray(visits).dtype))
    np.testing.assert_allclose(
        np.asarray(jnp.sum(out.action_weights, axis=-1)), 1.0, atol=1e-5)
    self.assertEqual(
        float(jnp.max(jnp.where(invalid, out.action_weights, 0.0))), 0.0)
    chosen_invalid = jnp.take_along_axis(invalid, out.action[:, None], axis=1)
    self.assertFalse(bool(jnp.any(chosen_invalid)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(
        out.search_tree.node_values))))

  @parameterized.parameters(1, 2, 3, 8)
  def test_chunking_is_exact(self, max_lanes_per_call):
    """`max_lanes_per_call` only bounds memory; it must not change results."""
    kwargs = _policy_kwargs()
    unchunked = mctx.batched_gumbel_muzero_policy(**kwargs)
    chunked = mctx.batched_gumbel_muzero_policy(
        max_lanes_per_call=max_lanes_per_call, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(unchunked.search_tree.summary().visit_counts),
        np.asarray(chunked.search_tree.summary().visit_counts))
    np.testing.assert_array_equal(np.asarray(unchunked.action),
                                  np.asarray(chunked.action))
    np.testing.assert_array_equal(np.asarray(unchunked.action_weights),
                                  np.asarray(chunked.action_weights))

  def test_zero_gumbel_scale(self):
    """gumbel_scale=0 makes many scores tie exactly; ties must stay valid.

    Used for evaluation on perfect-information games. With the Gumbel term gone
    the ordering is decided by logits plus completed Q alone, so this is the
    worst case for tie-breaking between `top_k` here and the sequential
    `argmax` in `gumbel_muzero_policy`.
    """
    kwargs = _policy_kwargs()
    # A uniform prior makes round 0 scores identical across all actions.
    kwargs["root"] = kwargs["root"].replace(
        prior_logits=jnp.zeros_like(kwargs["root"].prior_logits))
    got = mctx.batched_gumbel_muzero_policy(gumbel_scale=0.0, **kwargs)
    visits = got.search_tree.summary().visit_counts
    np.testing.assert_array_equal(
        np.asarray(jnp.sum(visits, axis=-1)),
        np.full((4,), kwargs["num_simulations"],
                dtype=np.asarray(visits).dtype))
    np.testing.assert_allclose(
        np.asarray(jnp.sum(got.action_weights, axis=-1)), 1.0, atol=1e-5)
    expected = mctx.gumbel_muzero_policy(
        gumbel_scale=0.0, qtransform=drift_free, **kwargs)
    got_exact = mctx.batched_gumbel_muzero_policy(
        gumbel_scale=0.0, qtransform=drift_free, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(expected.search_tree.summary().visit_counts),
        np.asarray(got_exact.search_tree.summary().visit_counts))

  def test_subset_score_matches_full_width(self):
    """The O(m) root scoring must equal mctx's O(num_actions) computation.

    `_completed_q_and_score_subset` re-derives the default qtransform's global
    terms (mixed value, rescale min/max, visit_scale, full-width logit max) from
    just the `considered` columns. No other test covers it: the drift-free case
    above passes a `functools.partial`, which fails the identity check that
    enables the subset path, and under the default transform a divergence from
    mctx is expected for unrelated reasons. So compare the two scorings directly
    on a tree that `gumbel_muzero_policy` itself built.
    """
    num_simulations, max_considered = 32, 8
    kwargs = _policy_kwargs(num_simulations=num_simulations,
                            max_num_considered_actions=max_considered)
    tree = mctx.gumbel_muzero_policy(**kwargs).search_tree
    visits = np.asarray(tree.children_visits[:, ROOT])
    # `considered` must hold visited actions (the function's precondition).
    order = np.argsort(-visits, axis=-1)[:, :max_considered]
    considered = jnp.asarray(np.sort(order, axis=-1))
    self.assertTrue(
        bool((np.take_along_axis(visits, np.asarray(considered), 1) > 0).all()))

    root_logits = tree.children_prior_logits[:, ROOT]
    gumbel = tree.extra_data.root_gumbel
    batch_range = jnp.arange(root_logits.shape[0])
    considered_prior = jnp.take_along_axis(
        jax.nn.softmax(root_logits, axis=-1), considered, axis=1)
    gumbel_sub = jnp.take_along_axis(gumbel, considered, axis=1)
    completed_full = jax.vmap(
        qtransforms.qtransform_completed_by_mix_value,
        in_axes=[0, None])(tree, ROOT)

    for considered_visit in (1, 2, 3, 4):
      scores_full = seq_halving.score_considered(
          considered_visit, gumbel, root_logits, completed_full,
          tree.children_visits[:, ROOT])
      expected = np.asarray(
          jnp.take_along_axis(scores_full, considered, axis=1))
      got = np.asarray(
          policies_batched._completed_q_and_score_subset(  # pylint: disable=protected-access
              tree, batch_range, considered, gumbel_sub, considered_prior,
              considered_visit, jnp.max(root_logits, axis=-1)))
      np.testing.assert_array_equal(np.isinf(expected), np.isinf(got))
      finite = np.isfinite(expected) & np.isfinite(got)
      if finite.any():
        np.testing.assert_allclose(expected[finite], got[finite], atol=1e-5)

  def test_non_default_loop_fn_is_rejected(self):
    """`loop_fn` has no meaning here; it must fail loudly, not silently."""
    with self.assertRaisesRegex(ValueError, "scan_fn"):
      mctx.batched_gumbel_muzero_policy(
          loop_fn=lambda *args: None, **_policy_kwargs())

  def test_custom_scan_fn_is_used(self):
    """A caller-supplied `scan_fn` (e.g. hk.scan) must be threaded through."""
    calls = []

    def counting_scan(*args, **kwargs):
      calls.append(1)
      return jax.lax.scan(*args, **kwargs)

    kwargs = _policy_kwargs()
    expected = mctx.batched_gumbel_muzero_policy(**kwargs)
    got = mctx.batched_gumbel_muzero_policy(scan_fn=counting_scan, **kwargs)
    self.assertNotEmpty(calls)
    np.testing.assert_array_equal(np.asarray(expected.action),
                                  np.asarray(got.action))


if __name__ == "__main__":
  absltest.main()
