# Derivatives and serialisation execution record

The user authorised autonomous implementation on 14 September 2026, with no
commits or pushes during the review. After reviewing the changes, they authorised
local commits for derivatives and serialisation; no push is authorised.
Both package overviews are required; serialisation stays short.
Changes outside the two packages are discouraged and tracked below for later review.
Later follow-up entries record subsequent design decisions. The latest serialisation
contract removes callable registration and stores declared model state only.

## Baseline

- HEAD: `be512cf`; main comparison: `828234e`.
- Existing source, tests, documentation, index diff, and working-tree diff captured
  at `/private/tmp/zodiax-autonomous-review-20260914` before this pass's edits.
- Existing changes to optimisation, its tests/docs, coverage, and general configuration
  belong to earlier work and must not be folded into this review.
- Derivative/serialisation tests were already moved into package directories. The
  existing pytest importlib setting permits matching filenames across directories.
- Baseline full-suite run with x64: **570 passed, 10 failed, 1 collection error**.
  The failures were within derivatives/serialisation: superseded dtype expectations,
  missing trainable-leaf checks, obsolete Mask storage/Mul usage, and a fixture that
  assigned the now read-only Mask.shape property during collection.

## Necessary changes outside package source and corresponding tests

| File | Reason | Verification or follow-up |
|---|---|---|
| `AGENTS.md` | Record the user's active no-commit instruction and scope. | Review-only guidance; no runtime test needed. |
| `REVIEW_PLAN.md` | Replace proposed commit gates with uncommitted review checkpoints. | Compare with the latest user instruction. |
| `REVIEW_LOG.md` | Keep this requested exception and verification record. | Updated as work completes. |
| `tests/derivatives/`, `tests/serialisation/` | Tests corresponding directly to the reviewed packages. | Final focused/default and full/x64 results below. |
| `docs/derivatives.md`, `docs/decompositions.md`, `docs/API/derivatives.md`, `docs/API/serialisation.md` | Keep package-facing documentation aligned with the two source overviews. | 24 runnable examples checked; isolated documentation build passed. |
| `docs/API/stateful.md` | Remove the stale callable-registration recommendation after the serialisation contract was narrowed. | Read-only audit found no remaining supported-callable claims; documentation build passed. |

No outside production-code changes have been made in this pass.
The pre-existing `pyproject.toml` importlib-mode change was included in the isolated
snapshot to support the already agreed test layout; its contents were not edited.
The pre-existing `mkdocs.yml` navigation was used for documentation verification and
was also left untouched. These earlier configuration changes remain for user review.

## Completed review phases

### 0. Baseline and integration

Captured the starting source, tests, docs, index and working diff before editing.
All package changes since main were reviewed, including the untracked decomposition
module. Retained the earlier test relocation. Removed the obsolete runtime-field
schema assumptions rather than restoring a deleted compatibility sentinel. The
final isolated snapshot imports successfully without unrelated dirty source files.

### 1. Derivative foundations

Reviewed `containers.py` and exports. Removed numerical archive hooks and their
duplicate storage validator. Kept explicit constructor assignments and useful shape
checks. Shared spectral diagnostics now use JAX; explicit `check()` performs the
eager scalar decisions. Layouts preserve ordered paths and shapes, including zero-sized
leaves. Fixed path-prefix validation when another name sorts between a parent and
its descendant. Module-qualified imports replace scattered method-local imports.

Realised inputs use the agreed `as_array(..., dtype=float)` policy. The differentiated
parameter tree has a separate contract: its leaves must already be floating, and
their dtypes are retained during differentiation. No parameters are silently filtered.

### 2. Derivative calculations

Reviewed `operations.py`, including the main-API Hessian conversion. Shared parameter
and function-output boundaries replace repeated checks and extra eager model calls.
One fixed-shape batch plan is reused. Legacy Hessian reconstruction now uses ordinary
Python layout slices and works inside JIT.

Gauss–Newton whitens before linearisation. A compiler optimisation barrier between
the forward and transposed whitened products is necessary: without it, XLA can
recombine factors into `1 / std**2`, reproducing overflow/underflow even after the
algebraic rewrite. This boundary does not perform validation or transfer arrays to
Python; it intentionally constrains compiler reassociation. Tests cover float32
scales `1e-20` and `1e20`, batching, JIT, and outer derivatives of point and noise.
No performance improvement is claimed or benchmark suite introduced.

`Jacobian.fisher` now computes covariance weighting with `J.T @ solve(C, J)` directly,
removing its dependency on the stats helper that constructs an inverse. `stats.py`
was not changed.

### 3. Decompositions and local coordinates

Reviewed all four classes in `decompositions.py`. Removed their archive hooks and
clarified array/tree types, stages, and constructor assumptions. Kept distinct
mathematical operations (`apply`, `solve`, `encode`, `pullback`, and natural-gradient
methods), with equations in the overview/reference rather than adding synonyms.

Projection solves now use the declared retained rank during SVD inversion; they do
not make a second tolerance-based rank decision. Zero-rank solves/encodings return
zeros, and out-of-range steps receive the documented least-squares representative.
Tests cover disparate parameter units, nonorthogonal retained columns, varying rank
inside a fixed JIT structure, and differentiation with respect to the supplied step.

Safe symmetric averaging and disabled duplicate JAX symmetrisation preserve large
finite float32 matrices such as `diag([2e38, 1])`. The same fix covers eigen/Cholesky
decomposition, projection, and spectral diagnostics. Empty layouts can be stored but
cannot be factorised. Structured uncertainty uses the row squared norms of the
projection rather than constructing the entire covariance just for its diagonal.

Mixed-dtype reconstruction was already corrected by conversion before ravel; retained
it and added a regression. A Module-origin regression verifies class, static metadata,
aliases, JIT results, unchanged origin values, and archive reconstruction.

### 4. Serialisation definitions and reconstruction

Reviewed `_leaves.py`, `_schema.py`, `_types.py`, `_callables.py`, `definition.py`,
`_reconstruction.py`, and `_validation.py`. Schema validation owns representation
checks; reconstruction consumes that validated data and checks installed-class
compatibility. Constructor-free allocation and field assignment remain local and
commented. No per-class hook or MRO validation framework remains.

Module path/alias rules no longer reject generic Base metadata with dotted keys.
The structural-template alias exception applies only to Module.alias, preserving
ordinary fields named alias on unrelated Equinox classes. Registered callables use
two simple lookup dictionaries under a plain lock; retained strong references make
the former repeated identity bookkeeping unnecessary.

NumPy remains only in the archive leaf codec for NPY headers, bounded host buffers,
and exact storage dtypes. Known-type resolution, explicit callable registration,
schema limits, static/dynamic field fidelity, typed PRNG reconstruction, and payload
checks remain. Unsupported dtype/configuration combinations raise rather than silently
narrowing loaded arrays.

### 5. Archive workflows

Reviewed `_archive.py`, public save/load, and exports. Kept atomic path replacement,
staged stream encoding, caller-owned handles, bounded payload readers, and complete
payload-consumption checks. Fixed silent success on short writes and writes returning
None without progress. Final stream-copy I/O failures can still leave partial stream
contents; this is explicitly separate from atomic filesystem replacement.

Rebuilt private-helper/corrupted-object tests around public representation and workflow
contracts. Tests include all three built-in typed-key implementations, bfloat16 and
complex arrays, empty/nonfinite array values, malformed schemas/framing/headers,
precision mismatches, failed writes, local classes, callable Modules, aliases, loaded
links and their accumulated gradients, State/StateRef, units, and derivatives.

### 6. Overviews and final verification

Completed the canonical derivative overview and short serialisation overview. Both
show actual object representations and runnable usage, current method diagrams, and
typed developer contracts mapped to package-structured tests. The derivative walkthrough
and serialisation API page include the corresponding canonical overview. The separate
decomposition reference explains equations and mathematical limitations. Test-file
references in the short overview use paths, avoiding links to nonexistent hosted tests.

## Verification results

Interpreter: `/Users/louis/mambaforge/envs/zodiax/bin/python` (Python 3.14.3,
JAX 0.7.2, Equinox 0.13.4). Commands use `PYTHONDONTWRITEBYTECODE=1`.

| Check | Result | Evidence |
|---|---|---|
| Baseline snapshot, `JAX_ENABLE_X64=True python -m pytest tests --continue-on-collection-errors -q` | 570 passed, 10 failed, 1 collection error | `/private/tmp/zodiax-autonomous-review-20260914/baseline-tests.txt` |
| Working tree, default precision, reviewed packages plus numerics/Base/Module | 646 passed before the last stream-write regression was added | `/private/tmp/zodiax-reviewed-default.txt` |
| Final working tree, `JAX_ENABLE_X64=True python -m pytest tests -q` | **664 passed** | `/private/tmp/zodiax-reviewed-full-x64.txt` |
| Final isolated snapshot, default precision, reviewed packages plus numerics/Base/Module | **647 passed** | `/private/tmp/zodiax-isolated-default.txt` |
| Final isolated snapshot, `JAX_ENABLE_X64=True python -m pytest tests -q` | **664 passed** | `/private/tmp/zodiax-isolated-full-x64.txt` |
| Black `--check` and Ruff on both packages and their tests; `git diff --check` | Passed | 23 Python files formatted; no lint or whitespace errors |
| Executable documentation | 22 derivative examples, 17 captured output blocks; 2 serialisation examples | Verified in fresh default-precision processes |
| Isolated `PYTHONPATH=src python -m zensical build --clean` | Passed, 1.92 seconds | `/private/tmp/zodiax-isolated-docs-build.txt` |

The isolated snapshot is `/private/tmp/zodiax-reviewed-source-20260914`: HEAD overlaid
with the two reviewed packages, corresponding tests/docs, and the earlier test-layout
configuration. Unrelated production code and tests, including optimisation, come
from HEAD. The snapshot includes currently untracked reviewed modules. Thus the
passing suite does not rely on pending optimisation edits or a compatibility shim.

The documentation build ran inside that snapshot with relative paths. An earlier
attempt with absolute output paths exposed a Zensical configuration-path panic;
the successful normal-layout build required no repository configuration change.

## API changes and remaining limits

- Relative to main, `jacobian` and `hessian` return structured objects instead of
  `(array, unravel)` tuples, and the implementation lives in `zodiax.derivatives`
  rather than `zodiax.diffops`. These existing development migrations are documented;
  `hessian_to_pytree` remains available. The initial review retained public names;
  the later serialisation follow-up removes `register_callable` as described below.
- Serialisation is new since main. Earlier development archives with runtime-field
  metadata are not a compatibility target; the final format records actual fields
  and their static status. There is no numerical class-validation callback contract.
- Eigenvector derivatives at repeated eigenvalues and rank-threshold crossings remain
  mathematically problematic. SVD-based solve derivatives with respect to the factor
  have the corresponding repeated-singular-value limit. Applying a fixed factor and
  solving with respect to its input retain their documented linear behaviour.
- Truncated covariance is an operator on retained modes, not evidence of zero physical
  uncertainty in discarded directions. Eigen and Cholesky damping conventions differ
  as documented. Direct projection constructors must provide independent retained
  columns and zero discarded columns.
- Archives preserve supported structure and values, not implicit Python identity,
  arbitrary constructor invariants, global unit settings, or device placement.
  Ordinary array allocation follows validated payload size; typed-key templates have
  a separate schema size bound. Existing format/resource limits remain explicit.

All in-scope failures are resolved. No outside production edits or unresolved
integration failures remain. Existing unrelated work is preserved. HEAD and the
index match the captured baseline: no commits, pushes, or staging occurred.

## Follow-up: shared class families and a simpler walkthrough

Added `Derivative` as the shared matrix/layout base of Jacobian, Hessian, and
GaussNewton. Hessian and GaussNewton retain their symmetric TreeMatrix behaviour;
Fisher remains outside the Derivative family. Removed forwarding constructors
whose only action was to call their parent constructor. Stored fields, shapes,
array conversion, and numerical calculations are unchanged.

Added `Decomposition` as the common parent of EigenDecomposition,
CholeskyDecomposition, ParameterProjection, and LocalParameterisation. It supplies
the common Base interface without requiring fields or methods that are not shared
by all four classes. Both new names are exported through the existing package and
top-level export lists.

Rebuilt the derivative walkthrough around one `Linear` Module, a parameter
dictionary, and named prediction, residuals, and loss functions. Jacobian, Hessian,
and GaussNewton appear together before factors and parameter steps. Later examples
reuse the same setup. TreeLayout appears in object representations, with its direct
API confined to the developer section. Updated the full inheritance diagram, typed
contracts, compact factor reference, and API page.

Eight additional tests cover family membership, inherited construction, immutable
updates, and archive round trips. The only source edits in this follow-up are in
`derivatives/containers.py` and `derivatives/decompositions.py`. Corresponding
derivative tests, the overview, `docs/decompositions.md`, and
`docs/API/derivatives.md` are the associated test/documentation changes; this log
records their verification. No outside production files were changed.

| Check | Result | Evidence |
|---|---|---|
| Derivatives and serialisation, default precision | **368 passed** | `/private/tmp/zodiax-hierarchy-default-tests.txt` |
| Full working-tree suite, `JAX_ENABLE_X64=True` | **672 passed** | `/private/tmp/zodiax-hierarchy-full-x64.txt` |
| Revised derivative overview | 10 executable blocks passed; 5 actual output blocks captured | Fresh default-precision execution |
| Class diagram | All 13 public classes and source inheritance edges match | AST comparison, including both Hessian/GaussNewton parents |
| Black, Ruff, and `git diff --check` | Passed | Derivative source and tests |
| Documentation build in refreshed isolated snapshot | Passed, 1.87 seconds | `/private/tmp/zodiax-hierarchy-docs-build.txt` |

The earlier verification table describes the initial review; this table records
the follow-up. HEAD and the index still match the pre-follow-up snapshot at
`/private/tmp/zodiax-hierarchy-review-20260914`. No commits, staging, or pushes.

## Follow-up: serialisation of declared model state

The user narrowed serialisation to a clear model-field contract. Removed the public
`register_callable` API, the entire `_callables.py` registry and automatic built-in
registrations, and the callable schema/reconstruction cases. There is no fallback
or migration route for earlier development archives containing callable nodes.
Those nodes now fail schema validation, including with `strict=False`.

Supported Zodiax/Equinox classes declare their persistent state in fields containing
supported arrays, literals, and containers. Callable Modules use the same contract:
their methods come from the installed class and are not saved or evaluated during
serialisation. Standalone functions, closures, bound methods, partials, and
function-valued fields are unsupported. This removes default Equinox MLP callable
support and stored JAX sampler function support. Ordinary field-based Equinox Modules
and the existing exact Equinox State codec remain supported.

The guarantee covers stored class/field structure and represented values under the
documented schema and environment requirements. It does not cover changed method
implementations, globals, or arbitrary constructor invariants. Replaced registration
tests with rejection boundaries and filename-based callable-model round trips,
including analytic gradients and JIT evaluation after loading.

The overview now starts with `save("model.zdx", model)` and `load("model.zdx")`,
followed by a small callable Linear Module example and a concise contract table.
Both examples were executed in a temporary directory with local-source `PYTHONPATH`;
their actual outputs are shown and no example archives remain in the repository.

Outside-package edits are limited to the stale serialisation paragraph in
`docs/API/stateful.md` and the implementation/review records (`AGENTS.md`,
`REVIEW_PLAN.md`, and this log). The guide and plan now reflect the new field contract.
No production code outside serialisation changed in this follow-up.

| Check | Result | Evidence |
|---|---|---|
| Serialisation suite, `JAX_ENABLE_X64=False` | **199 passed** | `/private/tmp/zodiax-serialisation-callable-scope-tests-20260914.txt` |
| Full suite, `JAX_ENABLE_X64=True` | **689 passed** | `/private/tmp/zodiax-field-contract-full-x64.txt` |
| Overview | Both filename examples passed; actual output captured | Local source, temporary working directory |
| Independent source audit | No stale registry paths; Module/Expression/Transform fields round-trip without method evaluation | Bounded save/load and restored-method probes |
| Black, Ruff, and `git diff --check` | Passed | Serialisation source and tests |
| Documentation build in refreshed isolated snapshot | Passed, 1.87 seconds | `/private/tmp/zodiax-field-contract-docs-build.txt` |

HEAD and the index match the pre-follow-up snapshot at
`/private/tmp/zodiax-serialisation-contract-20260914`. No commits, staging, or pushes.

## Approved local commits

After reviewing the completed work, the user authorised sensible local commits for
derivatives and serialisation. This supersedes the earlier instruction to leave
these changes uncommitted; pushing and publishing remain outside scope.

- `bc7f33e` — Simplify serialisation around declared model state. Includes its
  reorganised tests, filename-based overview, stateful documentation correction,
  and the pytest importlib setting needed for repeated package test filenames.
- `545580e` — Add reusable decompositions and simplify derivative calculations.
  Includes its reorganised tests, shared class families, overview/API pages,
  decomposition reference, and navigation entry.
- The shared implementation guide, review plan, and this execution record are
  recorded separately from the package implementations.

Before committing, built an isolated tree from `be512cf` plus exactly the selected
package source, tests, documentation, and necessary configuration changes. Pending
optimisation edits were excluded. All **689 tests passed** with `JAX_ENABLE_X64=True`
and the documentation build passed. Evidence is in
`/private/tmp/zodiax-commit-review-20260914/full-x64.txt` and `docs-build.txt`.
Black/Ruff and whitespace checks passed; both code commits also passed the installed
Black/Ruff commit hooks. A writable temporary pre-commit cache was used after the
default cache path was blocked by the workspace sandbox; no hooks were skipped.

The staged `.coverage` deletion and pending `.gitignore`, optimisation source,
optimisation tests, and optimisation documentation changes remain separate and
unchanged. No pushes were made.
