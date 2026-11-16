# Changelog

## 0.6.0

- Add `Module` for composing numerical models, with parameter aliases, named
  descendant lookup, and immutable multi-parameter updates.
- Add numerical expressions, transforms, array conversion, units, shared-value
  links, and state containers, with staged resolution and JAX transformations.
- Replace `diffops` with structured Jacobian, Hessian, Fisher, and covariance
  operations, including reusable matrix decompositions. Derivative results now
  expose named containers rather than the previous tuple-based interface.
- Add numerical modelling and derivative guides, API references, and tests for
  eager execution, JIT compilation, differentiation, and precision handling.

- Add versioned documentation, with a moving `main` version, release-specific
  archives, and a `latest` release alias. Existing documentation archives can be
  registered without rebuilding or overwriting them.
- Add automated releases from version tags, including version and main-branch
  checks, tests, distribution validation, PyPI publishing, and GitHub release notes.
- Build documentation for pull requests and expand the test matrix to Python
  3.10–3.14. Set a 95% project coverage target with informational patch coverage.
- Update package build configuration, licence metadata, project links, and
  development dependencies. Exclude generated coverage files and documentation
  build output from version control, and tidy source distribution contents.
- Update the contributing guide and installation instructions, expose the
  changelog and contributing guide in the documentation, and simplify API headings.
