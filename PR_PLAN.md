# Zodiax PR plan

Updated 16 November 2026. Local branch preparation only; no push or merge performed.

## Merge order and tracking

| Work | Branch / PR | Dependency | Status |
| --- | --- | --- | --- |
| Numerical foundation and derivatives | `numerics`, PR #89 | main | Serialisation extracted; local default/x64 tests pass |
| Serialisation | `serialisation` | numerical foundation | Preserved as a separate stacked branch |
| Optimisation compatibility | planned `optimisation-followup` | derivatives merged | Existing three-file working diff preserved on the local numerics checkout |
| Documentation and packaging | planned `docs-package` | final public API | Planned; mirror dLux overhaul practices |

Safety reference: `numerics-before-serialisation-split` preserves the original
committed tree at 769c981. The serialisation branch restores that implementation
on top of the split numerics branch. Existing numerics/derivatives documentation
and version metadata stay with the core; broader documentation changes are deferred.

## Core numerics PR

- [x] Retain Module, numerical expressions, transforms, units, links, state,
  derivatives and decompositions, their contracts and tests.
- [x] Remove serialisation source, exports, Base convenience methods, archive-only
  tests and documentation promises from the core tree.
- [x] Run committed-source tests in default and x64 precision.
- [ ] Push the split branch and check CI on its exact head before merge; resolve the Codecov status policy.
- [x] Review core changes from main, including the diffops replacement and public exports.
- [ ] Update PR #89 scope and merge after review.

## Serialisation PR

- [x] Preserve ObjectDefinition, archive I/O, schema/reconstruction and validation,
  Base.save/load, exports, documentation and all existing archive integration tests.
- [ ] After core merge, rebase onto main (if squash-merged, transplant only the
  serialisation restoration commit, not the full numerical history).
- [x] Run archive suites and full integration tests under default and x64 precision.
- [ ] Repeat remote CI after publishing/rebasing the branch.
- [ ] Review supported types, class resolution, template/version policy, failed
  writes and malformed archives; publish and merge the separate PR.

## Optimisation PR

Existing uncommitted files: src/zodiax/optimisation.py,
tests/test_optimisation.py and docs/optimisation_tools.md.

- [ ] Move these exact edits to a branch based on merged core; preserve their diff.
- [ ] Review eigen_projection delegation to Fisher.projection, mode ordering,
  covariance behaviour and rank/scale edge cases against the legacy contract.
- [ ] Run focused optimisation tests and full integration checks; document any
  intended behavioural change and submit the separate PR.

## Documentation and package PR

Reference: /Users/louis/Projects/Software/dLux/repo, overhaul branch.

- [ ] Audit public docstrings and execute examples against the final Zodiax API.
- [ ] Separate installation, tutorials, authored Module Overviews and API reference;
  define source ownership and adapt export-driven API/inheritance generation.
- [ ] Write a main-to-0.6 migration guide and replace the placeholder changelog.
- [ ] Adapt strict docs builds and generated-output consistency checks.
- [ ] Review licence metadata, supported Python versions and dependency requirements.
- [ ] Adapt dLux release validation: main ancestry, tag/version agreement, test
  matrix, wheel/sdist build and validation, PyPI publishing, versioned docs and
  release notes. Configure publishing separately before the first release.
- [ ] Validate built distributions and rendered documentation; verify dLux consumes
  the intended Zodiax release before coordinating publication.

Update this checklist with actual PR links and verification results as work lands.

## Split verification — 16 November 2026

Clean git-archive snapshots excluded the three uncommitted optimisation files.
Python 3.14.3, JAX 0.7.2, Equinox 0.13.4, local macOS CPU:

| Snapshot | Default precision | JAX_ENABLE_X64=True |
| --- | --- | --- |
| Core numerics (57e3151) | 517 passed | 517 passed |
| Serialisation restored (ab9a35e tree) | 719 passed | 719 passed |

The restored source/tests/docs match the pre-split committed tree exactly;
only this plan is additional. Core commit formatting/lint hooks passed. Remote CI,
documentation builds and package builds were not run in this split pass.

## Core merge-readiness review — 16 November 2026

Reviewed local head 8df281c against main 828234e. No new code blocker identified
in the reviewed resolution, links, Module lookup/update, transform, state, derivative
and decomposition paths. This is not a proof of correctness for every downstream use.
The source tree is unchanged since the split validation; subsequent commits update
only this plan. Uncommitted optimisation work remains excluded.

Additional verification:

- Built a 0.6.0 wheel with pip wheel --no-deps --no-build-isolation.
- Installed that wheel into an isolated temporary target and verified its import
  location; all 517 tests passed against that installed package, with pytest's
  source-path override disabled. Coverage: 1903/1955 statements, 97.34%.
- Checked that every public __all__ name exists. The new derivative return types
  and replacement of zodiax.diffops are documented in docs/API/derivatives.md.
- Documentation build completed with Zensical 0.0.24, but the tool warns that
  strict mode is unsupported. This is a build check, not validated strict/link QA.
- Git diff whitespace checks passed. No production code changes were needed.

Remote status inspected on PR #89: GitHub still has 769c981 (before extraction),
base 828234e, mergeable, no inline review threads and no PR description. Python
3.10–3.13 jobs pass. Codecov patch fails at 95.79% against a 100% target; project
fails at 96.17% against main's 100%. These results do not cover the local split.
Do not treat old green test jobs as approval of the current local head. Do not
silently weaken coverage targets; inspect the refreshed report and agree the policy.

Before merging:

1. Push numerics and preserve/publish serialisation separately as needed.
2. Update PR #89 to describe numerical model composition and structured derivatives,
   explicitly excluding serialisation and the pending optimisation changes.
3. Run the remote matrix on the split head and address its coverage status.
4. Merge only after those checks and review are accepted.

Deferred documentation issue: docs/optimisation_tools.md already imports the
nonexistent batching module on main and uses the old Hessian tuple API. The pending
optimisation edits touch that page; repair and execute those examples in the planned
follow-up. The separate API migration section describes the new contract now.
No push, merge, publication or coverage-configuration change was performed.
