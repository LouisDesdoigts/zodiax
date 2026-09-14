# Zodiax implementation guide

This is the working agreement for Zodiax implementation and review, adapted from
the dLux development guide. Update it as design decisions are agreed. The intended
reader is a new postgraduate student with basic Python experience.

## Working agreement

- Read the affected source, neighbouring modules, and call sites before proposing
  a new pattern. Establish the behaviour and public contract before changing APIs.
- Make the smallest coherent change, preserve unrelated work, and discuss broad
  redesigns before implementing them. Follow the user's latest direction.
- The current numerics pass includes the user overview, developer contracts, and
  numerical tests. Use those contracts to guide regression tests for public
  behaviour and promised JAX transformations. Keep serialisation and other package
  sections outside this pass; do not run their suites or rewrite their documentation.
- Preserve established Base behaviour on main. Review and change its new numerical
  features without redesigning existing path operations.
- Keep Module, aliases, descendant lookup, and multi-object updates in `module.py`.
  `base.py` owns the established path operations and retains the Equinox wrappers
  for their later review. Preserve their public top-level names and Base imports.
- Use ordinary attribute access first, local aliases second, and a breadth-first
  descendant search last. Examine every match at the nearest depth before choosing
  a value. Aliases are static, consistently ordered name/path pairs targeting
  stored dynamic fields. Explain the difference between these fixed paths and
  descendant-name lookup, and keep structural validation at explicit boundaries.
- Do not commit or publish externally without an explicit request.
- Judge unreleased development APIs against the agreed design and current source.
  Old migration snapshots alone do not justify compatibility machinery or dead code.

## Readable implementation

- Optimise for a student being able to read, explain, and update the implementation.
  Do not minimise line count at the expense of understanding.
- Lay out nontrivial functions as an algorithm: distinct blocks separated by
  whitespace, with short comments explaining each stage's purpose.
- Use ordinary loops and meaningful intermediate names when comprehensions or
  nested expressions conceal the steps. Prefer clarity in Python setup and tracing
  code over minor Python performance savings; consider array execution separately.
- Keep complete operations on one line when clear. Use deliberate multiline layouts
  and trailing commas when they reveal structure. Formatting does not define design.
- Use concise domain names without obscuring meaning. An alias should name a concept
  or clarify repeated access, rather than merely shorten one expression.
- An abstraction must represent a domain concept, public extension point, shared
  invariant, or meaningful reuse. Do not extract a helper just to shorten its caller.
- Give helpers distinct responsibilities. In resolution, separate tree traversal
  from handling one expression without scattering each condition into a new helper.
- Keep imports at module scope, functions together before classes, and constructors
  first among concrete class methods. Resolve ownership rather than hide import cycles.
- Assign fields visibly on `self` inside constructors. Equinox permits these initial
  assignments and freezes the completed object; avoid generic helpers that hide
  field assignment behind an `owner` argument and `setattr` loops.
- Keep `Transform` without a custom base constructor so Equinox can generate
  constructors for downstream subclasses that simply declare fields. Repeating
  the small `x` and `alias` assignments in built-in constructors is acceptable.
- Use `Map` for bias, scale, matrix contraction, and their fixed-order combination;
  do not retain separate `Add`, `Mul`, or `MatMul` classes. `Map.inv` provides its
  representative local reverse; `apply(..., inverse=True)` reverses the chain. Keep the
  `matmul` function for array and sampled-basis calculations.
- `Operation` is a `Transform` subclass for elementwise operations on any input
  shape. `Exp`, `Log`, `Pow`, and `Unit` use ordinary JAX broadcasting, including
  output shape expansion. Do not add a constructor or wrap subclass calculations
  to enforce this contract. Elementwise reversal does not undo broadcasting.
- Keep `Operation`, `Exp`, `Log`, and `Pow` in `numerics/operations.py`.
  `transforms.py` owns `Transform`, `Map`, `Mask`, and matrix contraction; units
  retain their catalogue and `Unit` class in `units.py`.
- Preserve Equinox's default object representation. Do not implement `__pdoc__`
  or inspect and hash array contents merely to customise printed summaries.
- Do not add `del` statements merely to mark unused method arguments. Keep the
  required signature and let the implementation express what it uses.

## Contracts and numerical behaviour

- Build a research toolkit with small, documented extension contracts and simple
  templates. Assume downstream authors follow those contracts; incorrect use or
  implementation may fail with ordinary Python or JAX errors.
- Do not automatically rewrite or wrap subclass methods to enforce base-class
  behaviour. Extra machinery must support a concrete intended workflow, rather than
  attempt to make arbitrary user implementations impossible to break.
- Keep validation purposeful: use clear construction boundaries and explicit
  diagnostics where useful. Avoid repeated hidden checks during numerical evaluation
  solely to defend against a caller violating the documented contract.
- Do not use `eqx.error_if` or equivalent runtime validation callbacks in numerical
  calculations. Invalid numerical inputs may follow ordinary JAX behaviour. Keep
  any useful validation at construction or archive boundaries, and distinguish
  Python shape checks during tracing from checks executed in the compiled model.
- Give array conversion and operand resolution clear owners. Convert inputs at the
  appropriate boundary, resolve required expression operands explicitly, and avoid
  repeatedly converting or resolving the same value through several wrappers.
  Keep these steps separate when nesting the calls would obscure the calculation.
- Use one `as_array(value, dtype=...)` conversion function. Use ordinary Python
  types (`float`, `int`, `complex`, `bool`) or `None` at conversion boundaries;
  let JAX's startup configuration choose default precision. Do not add explicit
  bit-width promotion or special handling for lower-precision numerical inputs.
  `dtype=None` preserves the inferred type, including integers and complex values.
  Real-valued derivative fields request `dtype=float`; generic transforms retain
  inferred types. Callers supply floating or complex trainable parameters.
  Preserve `None` and Expressions without attaching a future dtype conversion.
  Reusing an existing array's dtype for matching buffers or clearing weak typing
  is distinct from choosing a precision. Archive dtype metadata and the unsigned
  key representation required by JAX must retain their exact storage types.
  When creating PyTree reconstruction functions, convert leaves before flattening
  so reconstruction uses the same dtypes as the numerical calculation.
- Call `as_array` directly in numerical constructors; keep any required-value
  check visible in the owning constructor. Derivative-specific storage validation
  belongs with derivative containers. Use `eqx.is_array` for Python-side array
  detection and JAX operations for numerical work.
- Do not add per-class `__zodiax_validate__` archive hooks. Keep useful constructor
  checks visible; review generic archive structure and loading separately from
  numerical class implementations.
- Make inputs, outputs, shapes, dtypes, broadcasting, and relevant JAX behaviour
  explicit. Distinguish construction, validation, and numerical evaluation.
- Keep dependencies and changes in behaviour visible at call sites where practical.
  Prefer keyword arguments when they materially explain an option's meaning.
- Keep advanced Python mechanisms local and explain why they are needed, what state
  they carry, and how that state is restored when an operation fails.
- Retain `ContextVar` for nested expression bookkeeping and `inspect` for discovering
  required context. Explain their use beside the code, including the distinction
  between resolver bookkeeping and the scientific inputs in `**context`.
- Use one dedicated internal signal for expected incomplete evaluation. Explain where
  it is raised and caught; do not hide genuine programming errors behind that signal.
- The user-facing resolution operation is `self.resolve(**context)`. An ordinary
  Module contains numerical definitions: resolving it returns a prepared copy of
  the same class, replacing ready expression children with their numerical values.
- An Expression represents a value: resolving it returns that value when ready,
  or a partially prepared Expression when dependencies remain unavailable. A
  Transform is an Expression and resolves to its output array, including when
  resolved directly. This behaviour must not depend on having a parent container.
- Expression inherits Module for parameter storage and PyTree operations, while
  specialising resolution. Use this distinction before introducing another base
  class: an array-producing parametric is an Expression; an object that prepares
  its fields for a later model calculation is an ordinary Module.
- Prepare available descendants even when an enclosing expression lacks context.
  Retain the rebuilt expression with its prepared children until its remaining
  dependencies are available. Missing parent context must not discard child work.
- Keep `Expression.resolve_children` as the preparation method: it returns the
  expression with available children resolved, without computing its own value.
  Its inherited implementation recursively maps over dynamic child fields when
  the owner's calculation cannot finish. Implementing an expression does not
  require a custom preparation method.
- Determine readiness from required named context and operands actually requested
  by the calculation. Do not block evaluation just because an unused field or a
  stored callable transform remains an Expression. Ordinary Module operands may
  also retain definitions for their own later calls.
- Try `evaluate` before preparing children when its named context is available.
  This lets existing recipes supply local context inside their calculation. Missing
  context or an explicitly requested unresolved numerical operand falls back to
  child preparation, retaining whatever progress is available.
- Reuse completed expression results only within one outer resolution call and
  only for identical expression/context objects in the same JAX trace. Disable
  reuse across nested JAX transformations so inner tracers cannot escape through
  the cache. Keep reference lifetimes and cleanup explicit; do not retain cached
  arrays or tracers on the model.
- Design around staged model evaluation: a high-level resolve can turn mapped latent
  coefficients into coefficient arrays while retaining a coordinate-dependent OPD
  definition. The owning optic later resolves with wavefront coordinates, replacing
  its OPD child with an array while remaining an optic.
- Keep tree preparation distinct from expression computation in the implementation.
  An expression may need a prepared copy of its own fields before producing its
  value; that preparation must not recursively invoke its own value computation.
  Users should not need a package-level `zdx.resolve` function for normal model code.
- Keep `Expression.evaluate` as the method implementing an expression's numerical
  calculation. `resolve` coordinates traversal, link population, and readiness;
  it calls `evaluate` for ready expressions. Ordinary Modules do not need an
  `evaluate` method to prepare their fields. Transform already implements
  `evaluate` using its stored input and `fwd`; concrete transforms normally supply
  their mathematics through `fwd`. This distinction needs no method renaming.
- Declare a transform's required runtime context in its `fwd` signature. The
  inherited `Transform.evaluate` checks those requirements with the same signature
  inspection used for expressions, accounting for the positional `x` input.
  Missing context uses the existing incomplete-evaluation signal and child
  preparation. Do not require a forwarding `evaluate` override merely to repeat
  the context signature declared by `fwd`.
- Subclasses implement their local forward and reverse calculations in `fwd` and
  `inv`. `apply(value, inverse=False)` applies the complete nested Transform chain:
  deepest to outermost forwards, outermost to deepest backwards. Keep `inverse`
  a Python bool fixed during JAX tracing. A missing reverse raises NotImplementedError;
  Map and Mask retain their representative reverse rather than requiring bijections.
  Remove separate `decode`, `encode`, `solve`, and `project` transform methods.
  Retain `initialise` as an optional construction helper that binds a recovered
  deepest input using `apply(..., inverse=True)`.
- Transform constructors normalise stored numerical values while retaining
  Expression definitions. `evaluate` and `apply` prepare numerical
  inputs at their entry boundaries. `fwd` and `inv` accept and return
  arrays under ordinary JAX type promotion and resolve only the stored operands
  they consume. Generic conversion preserves integer and complex inputs; do not
  cast inverse solutions back to an integer input dtype. Numerical operand
  expressions must produce numerical scalars or arrays; do not wrap subclass
  methods to enforce these contracts.
- Use `x=None` in `Transform.__call__` to request stored-input resolution, including
  a partially resolved result. An explicit numerical input calls `apply` and may
  set `inverse=True`; an inverse call needs an explicit input value.
- Keep elementary arithmetic directly in the transform classes. Remove the public
  `add`, `mul`, and `power` wrappers. Retain `matmul` and `_solve_matmul` for their
  own review after this simplification pass.
- Store `Mask` selections as one dynamic boolean array with a static selected
  count. Derive temporary indices during forward and reverse evaluation using
  that fixed count; do not store indices or make the mask array static. Explain
  the resulting runtime cost in the class docstring. Updating mask values must
  preserve the selected count; do not add runtime checks for this contract.
  Construction needs a concrete mask because the selected count becomes static
  Python metadata; construct it before tracing the calculation.
- Store a Unit's input name in `unit` and its optional output name in `unit_out`;
  keep `to=` as the constructor keyword. `unit_out=None` follows the active global
  convention when evaluated; an explicit output name fixes the conversion.
  Globals are read during JAX tracing, so cached compiled calls retain that
  convention until retraced. Configure units before compiling model functions.
  Keep reference factors independent of these defaults.
  `Unit.to(unit, **context)` evaluates its stored input and returns a converted
  array without changing the original definition. `.to(None)` converts into the
  current global convention; missing input or required context raises ValueError.
  Use the constructor's `to=` argument for a definition with a fixed output unit.
  Build and validate replacement settings locally, then replace the global mapping
  once; return copies from the public settings functions. Unit scaling uses normal
  multiplication and division with ordinary dtype promotion, without special
  precision promotion or runtime numerical validation.
- Keep `State` as an ordinary Module using inherited `.set`; callers preserve
  array shapes, dtypes, and PyTree structure when using it as a JAX loop carry.
  Convert each named numerical entry to one JAX array at construction, including
  `index`. Scalar integer arrays support dynamic element indexing and indexed
  updates under JIT. Dynamic slice starts need JAX's fixed-size slicing operations;
  ordinary Python slice bounds must remain static. State holds current numerical
  values; expressions belong in the model.
  Use ordinary Module paths and aliases, including qualified paths for ambiguous
  names. `StateRef.evaluate` reads the numerical value directly; it needs no nested
  resolution under this storage contract. Missing state follows normal Expression
  deferral through the required `state` argument.
  `State.step` reads all inputs from the original state before applying its updates
  together. Keep `validate_state` as an optional check for missing referenced paths.
- Keep link population structural: collect all owners before replacing links,
  including nested owners and references that appear before their owners. Store
  each Linked owner in one tree position and use Deferred for additional uses.
  Retain duplicate-owner, missing-owner, and cycle diagnostics as Python topology
  checks. Keep collection, replacement, and their temporary dictionaries local to
  `populate`; reuse populated values by key within that call. `validate_links`
  shares this traversal without evaluating expressions. Linked may own an array,
  Expression, or PyTree; preserve containers and let callers supply their array
  leaves. Keep link keys visible through ordinary Equinox printing.
- Normal `resolve()` performs link population by default before evaluating ready
  expressions. Keep `populate()` as an advanced structural-only operation, allowing
  linked definitions to remain unevaluated for later local contexts. Lead user
  examples with `Map`, calls and resolution, units, and shared values before
  explaining developer methods or array-conversion details.
- Show real Equinox representations and numerical outputs in tutorials. Introduce
  units with `Unit(x=value, unit=...)` and `distance.resolve()`, relying on the
  global default convention. Give each numbered section a brief summary, include
  the current module tree and complete numerical class hierarchy, and express
  developer contracts as input/output type and shape tables with short examples.
- Components that change scientific context must control which context their
  children receive. Having coordinates available does not establish that they
  belong to the correct frame; signature inspection cannot determine this.
  Give distinct scientific inputs distinct context names: for example,
  `coordinates` for incoming coordinates and `sample_coordinates` for coordinates
  derived for a child. A context name must have one meaning throughout a tree;
  children consuming the same input can share its name. Nested wrappers must also
  distinguish different derived inputs instead of reusing one name for each frame.
  Keep this logic in `evaluate`; do not require a custom `resolve_children` merely
  to preserve normal evaluation order. The inherited fallback forwards the supplied
  context unchanged. Construction and caller contracts define valid combinations;
  do not infer scientific meaning or add machinery to protect unsupported inputs.
  Keep dLux edits out of this pass unless explicitly authorised.
- Keep `as_array` in `numerics`. Extend or
  move shared non-numerics functionality only when a concrete need justifies it.
- Use British scientific vocabulary in prose while preserving public identifier
  spelling unless an API change is explicitly agreed.

Before finishing a change, review the modified functions for visible stages, clear
ownership, accurate contracts, and consistency with the surrounding implementation.
