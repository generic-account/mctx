# Upstreaming notes — `batched_gumbel_muzero_policy`

Working notes for the fork. Not intended to be part of an upstream PR; delete
before proposing to `google-deepmind/mctx`.

## State

Branch `batched-gumbel-policy`, three commits off `upstream/main` @ `450fbf7`.
Fork PR: generic-account/mctx#1 (against this fork's `main`, as a review
surface — not a submission).

Gate, as of the last commit: **pylint 10.00/10 · flake8 0 · 41 tests · 0 lines
over 80 columns.**

## Dev environment

```bash
python3 -m venv /tmp/mctx-dev
/tmp/mctx-dev/bin/pip install -e ".[test]" pytest pytest-xdist flake8 pylint pylint-exit
```

Run their suite the way `test.sh` does — **from a directory inside the repo
root**, because `tree_test.py` opens its fixtures by relative path. Running from
anywhere else produces four spurious `FileNotFoundError`s:

```bash
mkdir -p _testing && cd _testing
/tmp/mctx-dev/bin/python -m pytest --numprocesses auto --pyargs mctx -q
cd .. && rm -rf _testing
```

Lint exactly as CI does (note `-efail -wfail -cfail`: conventions are fatal, and
their own files score 10.00):

```bash
/tmp/mctx-dev/bin/python -m pylint --rcfile=.pylintrc \
  $(find mctx -name '*.py' | grep -v 'test.py') -d R0917
/tmp/mctx-dev/bin/python -m pylint --rcfile=.pylintrc \
  $(find mctx -name '*_test.py') -d W0212,R0917
/tmp/mctx-dev/bin/python -m flake8 $(find mctx -name '*.py') \
  --count --select=E9,F63,F7,F82,E225,E251
```

House style that bites: **two-space indent**, 80 columns, `Args:`/`Returns:` on
private helpers too. `yapf --style='{based_on_style: google, indent_width: 2,
column_limit: 80}'` gets most of it but does **not** reindent docstring
interiors.

pytype is skipped on Python >= 3.12 — `test.sh` does this too, so a 3.14 venv is
fine.

## Upstream compatibility

Checked 2026-08: every difference in `_src/{search,action_selection,qtransforms,
tree}.py` between the pinned 0.0.71 release and `main` is a `# pyrefly: ignore`
comment. `seq_halving.py`, `base.py` and the public `__init__.py` are
byte-identical. The search algorithm has not changed semantically in ~2 years.

PyPI's newest mctx **is** 0.0.71 — the "Release v0.1.9" commit never published
(no git tag; a "Fix the pypi release hook" commit followed it). There is no
newer release to port to.

## Relationship to upstream PR #116

[#116](https://github.com/google-deepmind/mctx/pull/116) ("Avoid full-buffer
copies in the backward pass") independently arrived at the same path-record +
single-scatter technique used in `_backward_batched`, for the same reason. It
handles the **single-leaf** case, where each path node has one visitor so
`.set()` suffices; K lanes sharing the root need scatter-**adds** and the
closed-form root mean.

The two are **orthogonal and compose**: #116 removes per-simulation buffer
copies, this removes sequential `recurrent_fn` calls. Our only call into
`search.py` is `instantiate_tree_from_root`, and #116 touches only `backward`,
so there is no conflict and no reason to branch off it. The PR body should cite
it.

## Benchmarks

`tools/bench_mctx_policy.py` lives in the arimaa repo. mctx + jax only, so it
can be attached to an upstream issue unchanged.

```bash
python tools/bench_mctx_policy.py --n 200 --m 32 --batch 256 --reps 9 \
  --rec-cost cheap --variants mctx._src.policies_batched
```

`--variants` takes any module exporting `batched_gumbel_muzero_policy`, which is
how the edge-mirror designs were compared in one process. Measured results are
in `docs/MCTX_BATCHED_GUMBEL_PR.md` in the arimaa repo.

Watch out: `n=800` with `num_actions=1393` and `batch=256` OOMs a v5e chip
**inside `gumbel_muzero_policy`** — mctx's own `[B, num_nodes, num_actions]`
tables are ~6.9 GB at that shape. Not a property of this policy.

## Remaining before proposing upstream

- [ ] **Google CLA** (<https://cla.developers.google.com/>). One-time, per
      person, blocks any merge. Start early.
- [ ] **Open an issue first, not a PR.** The change is large and turns on a
      question only a maintainer can settle: is an opt-in policy with documented
      near-tie divergence acceptable, or must it be bitwise identical to
      `gumbel_muzero_policy`? Include the round table, the exactness argument,
      the benchmark, and a note that it composes with #116.
- [ ] Exercise `max_lanes_per_call` on hardware. Needs a shape where the tree
      fits but the widened `recurrent_fn` batch does not. Currently only proven
      bitwise-exact, not proven to relieve memory pressure.
- [ ] Finish the function-by-function pass. Done: `_rounds_from_schedule`,
      `_write_nodes_batched`, `_write_edge_record`. Not yet reviewed:
      `_backward_batched`, `_completed_q_and_score_subset`,
      `_make_forced_simulate`, `batched_gumbel_muzero_policy` itself.
- [ ] `vv_snap` in `_backward_batched` stacks `node_values` with
      `node_visits.astype(float32)` into one `[B, N, 2]` array — the same
      packing trick as the edge mirror, with the same questions. Safe as written
      (visits <= `num_simulations`), but it has not been through the same
      scrutiny or measurement.
- [ ] Decide whether the `action_selection.mask_invalid_actions` move ships as
      its own small PR first. It is self-contained (~30 lines, three files) and
      would de-risk the feature PR.
- [ ] Consider trimming the measured-rationale paragraphs in
      `_write_edge_record`'s docstring. They are useful to us; a maintainer may
      prefer them in the PR description instead.

## Design decisions already settled by measurement

Do not re-litigate these without new numbers — all were measured on a v5e-4 at
`num_actions=1393` with spreads <= 1.7%.

- **The edge mirror stays.** Removing it and reading `reward`/`discount` from
  the children tables costs **15–17%**, because every hop then double-indexes
  `[B, num_nodes, num_actions]`.
- **It packs all four values into one row.** Mirroring only the two floats (the
  indices are already in the tree, so no bitcasts would be needed) costs a
  further **2.8–3.9%**: three gathers per hop instead of one.
- **The mirror is int32, not float32.** Converting the indices numerically is
  silently lossy above 2**24, and `action` is bounded only by `num_actions`.
  int32 is also marginally faster.
- **Same-width rounds share one `scan`.** The Python `for` loop unrolled every
  round body: at `n=800, m=32` that was 404,906 StableHLO lines and 77.9 s to
  compile, versus 18,170 lines and 2.7 s grouped — and graph size now grows
  ~1.9x from n=32 to n=800 instead of ~33x.
- **Rejected: a `NamedTuple` to bundle `_write_nodes_batched`'s arguments.** mctx
  uses `NamedTuple` only for loop-carry state, disables `too-many-arguments` in
  `.pylintrc`, and has functions with 11–15 parameters. Many loose arguments
  *is* the house style.
