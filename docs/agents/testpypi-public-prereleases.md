# TestPyPI and Artifact Validation Policy for Bingo

## Recommendation

Use GitHub Actions artifacts as the normal release-validation mechanism. Do
not publish every branch or integration commit to TestPyPI. Publish intentional
public release candidates with unique `X.Y.ZrcN` tags to PyPI, including when
the tagged commit is on `develop`. TestPyPI is useful for occasionally
exercising the upload path, but it is not a faithful production rehearsal: it
is a separate, incomplete, periodically pruned index.

For a final release, build once on the exact release tag, verify that the
built-distribution version exactly equals that tag, validate the downloaded
artifact in clean environments, and publish that same artifact to PyPI only
after all validation succeeds. Use PyPI Trusted Publishing bound to the
release workflow; this avoids a long-lived upload token. PyPA's reference
workflow follows the same build, upload-artifact, download-artifact, and
tag-only publish shape.
Sources: <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>, <https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi>.

## Public Pre-releases and TestPyPI

**Benefits.** A public TestPyPI upload checks the external upload and simple
index path, gives collaborators a normal `pip` installation target, and can
provide an alpha or release candidate to users outside the repository. PyPA
explicitly presents TestPyPI as a separate PyPI instance for trying
distribution tools and processes without changing the real index, and its CI
guide describes TestPyPI artifacts as test builds for alpha users and as a
health check of the publishing pipeline. A real PyPI pre-release has the
additional benefit of resolving from the production index. Users normally have
to opt in: pip excludes pre-releases unless the requirement permits one or
`--pre` is supplied.

Sources: <https://packaging.python.org/en/latest/guides/using-testpypi/>, <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>, <https://pip.pypa.io/en/stable/cli/pip_install/#pre-release-versions>.

**Costs and risks.** Publishing makes a candidate available outside the CI
run, so its code, metadata, version, and support expectations should be ready
for public consumption. It also requires a unique version discipline and
operational credentials or Trusted Publishing configuration. TestPyPI accounts
and packages can be pruned, so it is not durable storage. PyPA warns that a
project which uploads on every frequent commit can exceed the project size
limit; it suggests a CI-local PyPI-compatible server instead. Finally,
pre-releases add user support and discoverability burden while pip will not
select them by default.

Sources: <https://packaging.python.org/en/latest/tutorials/packaging-projects/>, <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>, <https://test.pypi.org/help/#what-is-a-pre-release>.

## Is TestPyPI a Production Rehearsal?

Only partially. It is a good rehearsal for producing distributions, authenticating
to a Warehouse-compatible service, uploading them, and installing Bingo itself
from a remote simple index. It is not a reliable rehearsal for an ordinary
production `pip install bingo` dependency resolution:

- TestPyPI has a separate database and does not contain the same packages as
  PyPI. PyPA says dependencies can therefore fail to install or install
  unexpectedly, and recommends `--no-deps` for the basic TestPyPI installation
  example.
- Adding PyPI through `--extra-index-url` changes the candidate set. pip gives
  indexes no priority: it searches all configured locations and selects the
  best matching version. That is a different resolver input from a PyPI-only
  production install, and pip documents dependency-confusion risk for
  multi-index use.
- TestPyPI is periodically pruned. A passing result there cannot establish
  permanence, production availability, or production-index contents.

Sources: <https://packaging.python.org/en/latest/tutorials/packaging-projects/>, <https://packaging.python.org/en/latest/guides/using-testpypi/>, <https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages>, <https://pip.pypa.io/en/stable/cli/pip_install/#examples>.

## Proposed Bingo Workflow Policy

| Stage | Trigger and purpose | Required artifact policy |
| --- | --- | --- |
| Pull request CI | `pull_request` targeting integration branches; fast source test/lint/type/build checks. | Build distributions and run `twine check dist/*`; upload `dist/` for inspection or a downstream clean-install smoke job. Do not grant publishing permissions or upload publicly. |
| Integration validation | Pushes to `develop` and `main`; validate what will actually be installed. | Build Linux x86_64 wheels for Python 3.11 through 3.14 and an sdist, run `twine check`, upload immutable `dist/`, then download it in fresh jobs/venvs. Smoke-install every wheel and the sdist on Python 3.13 by file path; run the full installed-artifact suite for the Python 3.13 and 3.14 wheels. GitHub documents artifact sharing across dependent jobs and SHA-256 verification on download. |
| Release candidate | An explicit, uniquely versioned `X.Y.ZrcN` tag, including a tag on `develop`. | Run the complete installed-artifact matrix, then publish the tested files to real PyPI and create a signed GitHub prerelease for public feedback and production-index dependency behavior. |
| Production release | An exact final `X.Y.Z` tag (or a consistently named `vX.Y.Z` tag whose metadata is exactly `X.Y.Z`). | Build once, `twine check`, and run the full installed-artifact test matrix from the downloaded artifact. Publish the already tested files only after those jobs pass, with a tag-only PyPI job using `id-token: write` Trusted Publishing. |

Sources: <https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/>, <https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/storing-and-sharing-data-from-a-workflow>, <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>, <https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax>.

For Bingo's stated plan, the `main`/`develop` artifact build, `twine check`,
and clean wheel-install smoke check are the right baseline. The exact-tag full
installed-artifact suite is the release gate. Preserve the artifact across
jobs; rebuilding before publishing weakens the claim that the tested bits are
the published bits. A local file install tests the exact wheel without the
TestPyPI dependency/index distortion; a separate PyPI-only install after an
intentional public RC is the optional highest-fidelity consumer check.

Because there is no branch protection today, successful PR or branch workflows
are informative but do not enforce that `main`/`develop` received a passing
change. Bingo treats pushing an eligible release tag as the explicit publication
authorization and does not require a manual GitHub Environment approval. Keep
the publishing workflow's permissions minimal and restrict its trigger to
release tags; Trusted Publishing needs `id-token: write`, not a stored PyPI
token.

Sources: <https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi>, <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>.

## Open Decisions and Uncertainties

- This note does not inspect a proposed workflow, Bingo's version source, or
  the desired Python/OS matrix. The exact tag glob, artifact names, smoke
  command, and full installed-artifact suite still need to be specified
  against those details.
- The primary sources endorse both per-main TestPyPI publishing and tag-only
  PyPI publishing; they do not prescribe a universal cadence. The
  artifact-first recommendation is a tradeoff for Bingo's stated desire to
  validate artifacts without treating every integration build as public.
- A TestPyPI upload may still be worth retaining as a separately approved,
  low-frequency pipeline probe. It should not be treated as release approval,
  and the workflow should not combine indexes unless that mixed-index behavior
  is the thing being tested.
- If public RC feedback, third-party installation instructions, or download
  telemetry becomes a release goal, publish uniquely versioned RCs to PyPI
  rather than relying on TestPyPI. PyPI filenames cannot be reused, even after
  deletion, so release candidates must be immutable and uniquely numbered.

Source: <https://pypi.org/help/#file-name-reuse>.