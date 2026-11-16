# Pre-numerics publication plan

1. Merge docs-package into main first. It preserves version 0.5.0 and changes no
   production source or tests. Normal documentation pushes update /main/ only.
2. Run Archive current main documentation for 0.5.0. The live site already has
   /0.5.0/ and /v0.5.0/, but versions.json currently lists only 0.4.1. Register the
   existing archive without replacing its pages and set it as the default.
3. Verify the historical archive and version selector on GitHub Pages.
4. Bring these workflow changes into numerics and reconcile numerics-docs-package,
   which contains the new API docs/migration material. Merge numerical changes only
   after the historical archive has been verified.
5. Keep serialisation and optimisation as the previously planned follow-up PRs.

Workflow parity with dLux: Python 3.10–3.14 test matrix, 95% local coverage floor,
coverage artefacts/Codecov, generated API checks, tagged-release ancestry/version
checks, tests before wheel/sdist validation and PyPI publishing, versioned docs,
latest/default selection, and changelog-derived GitHub release notes. Zodiax adds
PR docs builds and explicit local link checks. Its moving channel is named main;
dLux currently still names that channel dev. No dLux files changed in this pass.
Existing historical archives are retained; this is not an automatic rebuild of all
old Git tags. PyPI trusted publishing configuration still requires verification.

Local validation: 88 tests pass, 100% statement coverage; wheel builds; docs build,
generated-reference check and local page/anchor checks pass. Zensical 0.0.24 still
warns that its strict flag is unsupported. Workflow YAML parses. Hosted workflow
execution, publishing and the historical index migration have not been run.
