# Derivatives and serialisation review plan

Status: reviewed autonomously on 14 September 2026. After reviewing the changes,
the user authorised local commits for derivatives and serialisation, with their
tests, documentation, and review records. No pushes or publishing are authorised.
Both overviews are complete, with serialisation kept short. Necessary changes
outside the two packages and their tests are recorded in [REVIEW_LOG.md](REVIEW_LOG.md).

The objective is to finish the existing development work so that its API,
implementation, examples, and tests agree. The implementation should be readable
and extensible by a new postgraduate student with basic Python experience.

## 1. What the numerics review established

We reviewed actual use cases and each file's responsibilities before changing the
code. Small examples exposed ambiguities in the API; implementation changes then
followed the agreed behaviour. We simplified structure and consolidated repeated
concepts, while retaining machinery that served a demonstrated purpose.

| Area | Result of the review |
|---|---|
| Resolution | An ordinary Module resolves to a prepared copy of its own class. An Expression resolves to its value when ready, or retains a partially prepared definition. Available child work is preserved; resolved values can also be supported PyTrees or Modules. |
| Extension | An Expression implements `evaluate`; a Transform normally implements `fwd` and optionally `inv`. Readiness follows required context and operands actually used. Ordinary subclasses need no custom child traversal. |
| Context | Scientific inputs remain explicit arguments, with distinct names for distinct coordinate frames. Signature inspection and temporary ContextVar bookkeeping remain because they support staged evaluation, nested calls, and JAX tracing; their purpose and lifetime are explained locally. |
| Transform API | Map combines scale, matrix or basis contraction, and bias. Redundant arithmetic classes and functions were removed. `apply(value, inverse=True)` replaces competing chain-reversal methods; local calculations remain `fwd` and `inv`. Representative reverses are allowed and documented. |
| Module boundaries | Elementwise Operations have their own file; Exp, Log, Pow, and Unit share that interface and ordinary JAX broadcasting. Domain-specific profiles and bases are downstream Expressions. |
| Arrays and masks | One `as_array` function owns conversion, preserving optional None and Expression inputs. JAX configuration owns default precision. Mask stores one dynamic boolean mask and a static selected count; its temporary index calculation and runtime cost are explicit. |
| Units and state | Unit keeps an optional output convention and `.to(...)` returns an array. State holds JAX arrays, including an integer-array index, and uses inherited immutable updates. Global units and fixed-shape loop state have explicit tracing contracts. |
| Sharing | Link population has a visible collect-then-replace algorithm with local bookkeeping. Normal resolution populates links automatically; structural-only population remains available. |
| Base and Module | Alias and model lookup logic moved into `module.py` and was simplified and commented. Established Base operations were preserved; Equinox wrappers remain in `base.py` for a later review. |
| Implementation | Visible constructor assignments replaced generic field-setting helpers. Scattered wrappers, redundant conversions, unused sentinels, `del` statements, pdoc machinery, and per-class archive validation hooks were removed from numerics. Useful shape and topology checks remain. |
| Evidence | The overview now starts with realistic usage and actual object representations, then explains extension. Typed developer contracts drive focused public-behaviour tests, including relevant JIT, differentiation, batching, and state workflows. |

The outcome was fewer competing ways to do the same thing, clearer ownership of
each operation, and explicit behaviour at the boundaries. Line count and test count
were evidence of changes, not targets.

## 2. Rules carried into the next review

The detailed implementation agreement remains in [AGENTS.md](AGENTS.md). Apply these
rules consistently in both packages:

1. Establish the public contract and scientific meaning before refactoring. Read
   neighbouring code, tests, documentation, and consumers as well as the file itself.
2. Prefer visible algorithm stages, ordinary loops, descriptive intermediate names,
   and concise comments explaining why. Type hints should describe actual supported
   inputs and outputs; distinguish constructor inputs from stored array fields.
3. Give each conversion, traversal, calculation, and validation a clear owner. Keep
   helpers for meaningful reuse or a distinct responsibility, not merely a shorter
   caller. Split files by responsibility only when that makes the dependency graph
   and reading order simpler.
4. Use ordinary assignments during construction and immutable updates afterwards.
   Explain necessary advanced mechanisms at their point of use. Archive rebuilding
   may need constructor-free allocation and field assignment: keep that exception
   local and justified rather than routing ordinary constructors through it.
5. Keep Equinox printing, simple subclass templates, and explicit dependencies. Do
   not add pdoc customisation, subclass rewriting, generic method interception, or
   hidden field-assignment frameworks.
6. Do not add `eqx.error_if` or numerical runtime validation callbacks. Keep useful
   structural checks at construction and optional diagnostics explicit. Document
   numerical domains instead of attempting to prevent every incorrect use.
7. Use JAX for numerical work and the shared array conversion policy. Keep dtype
   reconstruction consistent with the arrays being flattened. Host NumPy use in
   archive encoding, dtype inspection, or byte I/O requires a concrete explanation;
   replacing it with JAX is not automatically a simplification.
8. Remove per-class `__zodiax_validate__` plumbing and review validation ownership
   centrally. Archive/schema/payload integrity checks have a different purpose from
   numerical value checks; do not discard them mechanically or replace removed hooks
   with a registry of every numerical class's private invariants.
9. Consolidate genuinely equivalent APIs. Preserve distinct mathematical operations
   and established terminology when it helps users. For example, a linear-system
   solve and a coordinate map need not share the same method name just because both
   involve matrix multiplication.
10. Judge new APIs against current use cases, not earlier development snapshots.
    Record changes from main. Do not add compatibility layers solely for unreleased
    intermediate class layouts, method names, or archives.

## 3. Scope and autonomy

In scope are `src/zodiax/derivatives/`, `src/zodiax/serialisation/`, their exports,
overviews and API pages, corresponding test directories, and directly necessary
integration fixes. This includes completing existing uncommitted work in those
areas. Preserve unrelated edits and the staged coverage-file deletion.

Optimisation, stats, dLux, and a redesign of Base's established operations or Equinox
wrappers are outside this pass. Inspect consumers to understand a change; do not
silently widen the implementation task to those packages. The agreed numerics API
is a regression boundary, not another redesign target.

Executing this plan includes routine internal consolidation, fixes within the
documented mathematical contract, comments, tests, and documentation. The initial
implementation remained uncommitted for review; local package commits are now
authorised. Do not stop for approval after each file. Avoid
changes outside derivatives and serialisation; log necessary exceptions and their
tests for later review.

Pause only for a decision that the agreed contracts cannot settle: an additional
breaking change to an established main API or the agreed numerics API; incompatible
scientific interpretations with no supported default; a loss of intended archive
fidelity; or a necessary expansion into an excluded component. Prepare a concrete
example, recommendation, and impact before asking, and continue independent work.
Routine implementation choices and ordinary test failures are not pause conditions.

## 4. Required loop for every file or coherent group

Complete this sequence before moving on:

1. **Read and inventory.** Describe the public responsibility, stored state, callers,
   helpers, validation, and new behaviour relative to main. Trace conversion and
   traversal through the complete call path to find duplication.
2. **Write the contract.** Add or refine a short table of input types/shapes, output
   types/shapes, scientific meaning, supported transformations, and unsupported
   cases. Use a small runnable example when behaviour could be ambiguous.
3. **Decide what to keep, simplify, combine, move, or remove.** Record meaningful
   API changes and why each retained advanced mechanism is needed. Do not invent
   extra capability just to justify an abstraction.
4. **Implement.** Preserve the scientific contract while making the algorithm and
   ownership visible. Review the result from a student's perspective: can a reader
   explain the calculation without chasing a chain of tiny helpers?
5. **Verify against the contract.** Write or rebuild necessary public-behaviour tests
   and reproduce relevant bugs. Use independent mathematical references where
   possible. Update executable examples with the final API and actual output.
6. **Review and checkpoint.** Inspect the complete diff and integration, run the
   appropriate checks, and leave a coherent verified diff. Record changes,
   evidence, and remaining limitations in this plan's execution log.

Independent read-only reviews and bounded test work may run in parallel. Keep edits
to shared APIs and dependent files coordinated. Report progress
and findings without turning every update into a permission request.

## 5. Execution phases

### Phase 0 — Baseline and test layout

- Record HEAD, the main comparison point, staged changes, untracked files, and
  existing failures. Protect the current work before reorganising it.
  Review the complete new packages relative to main, including already committed
  development code and untracked modules; the remaining dirty diff is not the scope.
- Finish the already prepared test moves into `tests/derivatives/` and
  `tests/serialisation/`, retaining root tests for top-level modules. Keep pytest's
  importlib mode so matching test filenames do not collide.
- Compare main's `diffops.py` with the current derivative API. Main has neither the
  new derivatives package nor serialisation; existing derivative return-type and
  module-path changes need an explicit migration inventory, not accidental removal
  or restoration during cleanup.
- Repair the committed-source import dependency on the removed `_RUNTIME_FIELD`
  using the current design. The working tree currently masks this problem with
  pending serialisation edits. Do not introduce a legacy sentinel to hide it.
- Update obsolete serialisation fixture construction sufficiently to restore test
  collection. The current `_unsafe_module` fixture tries to assign `Mask.shape`,
  which is now a property. This is a collection-time fixture failure, not evidence
  of a loading defect. Replace old stored-index payload expectations as well; do not
  restore obsolete production fields to satisfy them.

Exit: imports and collection are usable; pre-existing failures and migrations are
recorded; test movement and necessary integration repairs can stand as coherent
review checkpoints independently of unfinished implementation.

### Phase 1 — Derivative layouts and containers

Review `containers.py` in conceptual order: TreeLayout; TreeVector and TreeMatrix;
Jacobian; Hessian, GaussNewton, and Fisher; explicit diagnostics.

- Establish path ordering, flattening and reconstruction, static metadata, selected
  coordinates, shape rules, and parameter dtype policy.
- Distinguish realised data, which constructors can convert with `dtype=float`, from
  trainable parameter trees, whose callers supply floating leaves. Align weighting
  inputs and tests with that distinction; do not silently convert trainable integers.
- Review repeated constructor/archive validation, array conversion, local NumPy
  use, inheritance, and the cost of eager diagnostics versus numerical methods.
- Keep dense result arrays separate from temporary differentiation closures.
- Establish the equations and noise conventions of `Jacobian.fisher` alongside the
  containers. Keep inverse covariance, covariance, and standard deviation distinct.
- Consider moving layout code only after checking whether that clarifies ownership;
  a split is not a required outcome.

Exit: layouts reconstruct and index the intended coordinates; container structure
and immutable updates are clear; contracts and focused tests agree.

### Phase 2 — Derivative calculations

Review `operations.py`: Jacobian, exact Hessian, Gauss–Newton, shared column
materialisation/batching, and the existing Hessian conversion compatibility path.

- Show flatten → linearise/differentiate → batch/materialise → wrap as visible steps.
- Specify scalar objective versus unreduced residual output and parameter-axis order.
- Check both Hessian methods, batching including partial final batches, checkpoint
  options, and JIT behaviour using independent derivative references.
- Fix avoidable overflow/underflow in standard-deviation weighting without inserting
  runtime guards or changing the mathematical noise model.
- Keep any required main-API migration support deliberate and local; remove only
  development-snapshot compatibility without further design work.

Exit: Jacobian and Hessian match reference derivatives; weighted Gauss–Newton agrees
with the corresponding explicit Jacobian formula, including extreme finite scales;
batching and execution options preserve results.

### Phase 3 — Decompositions and parameter coordinates

Review `decompositions.py`: eigen and Cholesky decompositions; ParameterProjection;
LocalParameterisation; their methods on derivative containers.

- Explain each calculation with its equation before reconsidering names such as
  `apply`, `solve`, `encode`, `pullback`, and the natural-gradient methods. Combine
  duplicate operations; retain methods that represent distinct useful calculations.
- Make equilibration, damping, retained-rank decisions, and reconstruction visible.
  Explain the existing method-dependent damping conventions. Specify empty-layout
  support explicitly and distinguish discarded covariance modes from a claim of zero
  statistical uncertainty in unconstrained directions.
- Reproduce and fix the projection solve that applies a second rank cutoff and the
  zero-rank NaN case. Respect the original retained-mode decision.
- Verify mixed-dtype origin reconstruction; earlier findings may already be addressed
  by the current conversion policy, so reproduce before changing it again.
- Define differentiability limits at changing rank and repeated eigenvalues.
  Distinguish a smooth covariance calculation from a potentially undefined derivative
  of its chosen eigenvectors, and applying an existing projection from differentiating
  its construction. Do not claim unrestricted differentiability.

Exit: reconstruction, solve, whitening/projection, covariance, and coordinate-mapping
identities hold on their supported domains; rank-zero behaviour is explicit and
finite where promised; numerical limitations are accurately documented and tested.

### Phase 4 — Serialisation definition and reconstruction

Start with the `save`/`load` workflow and ObjectDefinition contract, then trace their
implementation from `_leaves.py` and `_schema.py`, through `_types.py`,
`definition.py`, `_reconstruction.py`, and `_validation.py`.

- Make the stages visible: describe a tree, validate its schema, resolve known types,
  build a structural template, restore payload values, check topology.
- Establish the supported object/leaf table: Modules, containers and keys, static
  metadata, None, Python literals, concrete JAX arrays, and PRNG keys. Callable
  Modules store declared fields; functions and function-valued fields are unsupported.
  Keep explicit rejections understandable; do not expand supported types
  merely to make a generic recursion handle everything.
  Fidelity covers recorded structure and values, not arbitrary repeated Python
  object identity. Sharing uses the explicit Linked/Deferred contract after load.
- Explain why loading does not call user constructors, and retain a small downstream
  extension contract without requiring numerical classes to implement archive hooks.
- Check the scope of alias and dotted-path validation: preserve supported generic
  Base metadata rather than blanket-applying Module-specific path restrictions.
- Consolidate duplicate traversal and validation only where their semantics match.
  Parameter-layout traversal and complete archive traversal describe different trees.
- Preserve explicit class resolution without executing imports named by an
  archive. Keep shape, dtype, static/dynamic field, and full-payload consistency checks
  at the boundary where they are needed.
- Review `like`, `custom_types`, strict package-version checks, and format-version
  handling as separate contracts. There is no serialisation format on main; no
  migration engine is required for earlier development archives. Keep lossless
  representation and document the final supported format.

Exit: a reader can follow reconstruction end to end; supported structure and values
round-trip; type resolution and callable rejection behaviour are explicit; no
per-class numerical archive-hook framework remains.

### Phase 5 — Archive I/O and end-to-end persistence

Review `_archive.py`, complete `serialisation.py`, and check the public exports and
existing Base save/load conveniences.

- Follow path/file handling → definition/manifest → array payload → final write;
  make the corresponding load sequence just as visible.
- Keep path replacement atomic and preserve caller-owned file handles. For file
  objects, distinguish failures before the staged archive is copied from I/O failures
  during copying; an arbitrary stream cannot promise atomic replacement. Check
  malformed/truncated/mismatched payloads without giant test allocations. Explain
  any retained reader/writer wrapper by its job.
- Verify numerical models with Map, Mask, units, links, aliases, State and StateRef;
  also derivative containers and decompositions. Persist definitions without resolving
  them, then compare the loaded object's behaviour with the original.
- Verify exact supported dtype and PRNG representation. A process configuration that
  cannot represent a stored dtype must follow an explicit policy, not silently lose
  precision. Archive fidelity is separate from numerical default precision.

Exit: public save/load and template loading work for supported scientific models;
failure handling preserves the intended file and object contracts; implementation
has no obsolete numerical-field assumptions.

### Phase 6 — Documentation, regression checks, and final review

- Finish each package's user overview in the numerics style: immediate useful
  examples, real object printouts and outputs, brief section summaries, accurate
  package tree, and complete relevant class/method diagrams. Keep the serialisation
  overview short.
- Put typed developer contract tables at the bottom, linked to corresponding test
  files. Include a small downstream extension example where useful.
- Update API pages and exports to match; remove stale documentation rather than
  maintain competing descriptions of old and new APIs.
- Run the final verification matrix below and inspect an isolated source snapshot
  built from HEAD plus the intended package changes, including untracked modules.
  Verify that unrelated dirty production files are not hiding a missing dependency.
- Report reviewed files, API changes, important fixes, verification results, limitations,
  and any unrelated outstanding work. Do not describe a release as complete or
  publish it as part of this review.

## 6. Verification and review gates

| Area | Required evidence |
|---|---|
| Layouts and containers | Stable coordinate order; flatten/reconstruction and block identities; scalar, vector and mixed-shape trees; supported dtype conversion; immutable results. |
| Derivative calculations | Analytic or direct JAX Jacobian/Hessian reference; weighted `J.T @ W @ J`; both Hessian methods; batching equivalence; applicable JIT and differentiation workflows. |
| Decompositions | Reconstructed matrix; solve residuals; covariance/projection identities; full, deficient and zero rank; disparate physical scales; uncertainty tree shape/dtype; documented spectral differentiation boundaries. |
| Serialisation | Supported-value and structure round-trips; installed-class behaviour after load; type resolution and callable rejection; template/version policy; keys and exact dtype; meaningful malformed-archive and failed-write cases. |
| Numerics integration | Existing numerics and Base/Module regression suites; staged resolution, aliases and links survive supported persistence; derivative calculations operate on numerical definitions inside the differentiated function. |
| Documentation | Executed examples and printed outputs; current exports/diagrams; every promised developer contract represented by an appropriate test or a clearly stated unsupported case. |
| Final source | Formatting/lint and diff checks; package import, collection and relevant tests from an isolated source snapshot, without relying on unrelated dirty files. |

Run focused tests after each coherent change. Run affected suites under default
precision and with `JAX_ENABLE_X64=True` in separate Python processes at package
completion. Use dtype-appropriate tolerances; do not hard-code a global precision
policy into production code to make tests pass. Test only the transformations that
an operation promises; saving/loading concrete arrays is a host-side operation.

Rebuild tests from the contracts and retain useful existing regressions. Avoid
private-helper assertions, exact full repr or exception prose, exhaustive invalid
subclass scenarios, and tests that merely restate the implementation. Archive
corruption fixtures must target actual stored-format contracts. No coverage target
or benchmark suite is required; measure a specific runtime tradeoff only when it
needs evidence.

Run the full repository suite once at the final integration gate. Record any
remaining out-of-scope failures against the baseline; do not silently fix or commit
optimisation work to make this pass. Resolve all failures introduced by this review
and all pre-existing failures within the reviewed scope. Replace obsolete expectations
with tests of the agreed contracts rather than preserving superseded implementation.

Leave coherent reviewed units with their necessary tests, contracts, exports, and
callers. Commit only the authorised package work and associated review records;
preserve unrelated staged changes. Review each package diff and record verification in
REVIEW_LOG.md. Necessary out-of-package edits need their reason and test results
recorded there for the user to inspect later.

## 7. Execution record

Planning baseline: main `828234e`, HEAD `be512cf`, plus the existing working-tree
changes. The planning audits read source and history; they did not rerun tests.
Previously reproduced numerical findings must be checked against current source
during their assigned phase before being reported as current defects.

- [x] Phase 0: baseline, imports, collection, and test layout.
- [x] Phase 1: derivative layouts and containers.
- [x] Phase 2: derivative calculations.
- [x] Phase 3: decompositions and parameter coordinates.
- [x] Phase 4: serialisation definition and reconstruction.
- [x] Phase 5: archive I/O and persistence integration.
- [x] Phase 6: final documentation, verification, and uncommitted diff audit.
- [x] User-approved local package commits, preserving unrelated work.

For each completed phase, append a short record of decisions, changed public
behaviour, checks and results, reviewed files, and remaining issues. Mark a phase
complete only when its exit criteria hold. Keep this file current so execution can
resume without repeating reviews or losing unresolved findings.

Completed phase decisions, verification evidence, remaining mathematical limits, and
all necessary changes outside package source are recorded in [REVIEW_LOG.md](REVIEW_LOG.md).
The reviewed changes are approved for local commits; publication remains outside scope.
