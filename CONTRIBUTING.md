# Contributing Guidelines

We're posting this blueprint on GitHub to support the NVIDIA LLM community and facilitate feedback.
We invite contributions!

Use the following guidelines to contribute to this project.


## Pull Requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the upstream this repository.
2. Git clone the forked repository and push changes to the personal fork.
3. Once the code changes are staged on the fork and ready for review, a Pull Request (PR) can be requested to merge the changes from a branch of the fork into the `develop` branch of the upstream repository.
   - **Important**: All PRs must target the `develop` branch, not `main` or `master`.
   - The `develop` branch serves as the integration branch for ongoing development.
4. The PR will be automatically validated by our CI/CD pipeline (see Continuous Integration section below).
5. The PR will be accepted and the corresponding issue closed only after:
   - All CI checks pass successfully
   - Code review is completed and approved
   - Adequate testing has been verified


## Code Quality and Linting

Before submitting your pull request, please ensure your code follows our quality standards. We use Ruff and Pylint for code quality checks.

For detailed information on setting up and running linting tools, please see our [Linting Guidelines](LINTING.md).

**Quick setup:**
```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```


## Continuous Integration

This repository uses GitHub Actions for automated CI/CD. The following workflows are configured:

### CI Pipeline (`.github/workflows/ci-pipeline.yml`)

The CI pipeline runs automatically on:
- **Pull Requests**: Validates all changes before merge
- **Scheduled Runs**: Nightly at 7:30 PM UTC (1:00 AM IST)
- **Manual Trigger**: Can be triggered via workflow_dispatch

**Checks performed:**
- **Helm Blueprint Compliance**: Validates Helm chart configurations
- **Linting**: Runs pre-commit hooks including Ruff and Pylint
- **Unit Tests**: Executes Python unit tests with coverage reporting
  - See [Unit Test Documentation](tests/unit/ci-unittest.md) for running these tests locally
- **Frontend Unit Tests**: Runs frontend test suite with coverage
- **Markdown Link Validation**: Checks for broken links in documentation
- **Integration Tests**: Comprehensive integration testing including:
  - Basic integration tests
  - Reflection tests
  - NeMo Guardrails tests
  - Image captioning tests
  - VLM generation tests
  - Custom prompt tests
  - Observability tests
  - See [Integration Test Documentation](tests/integration/README.md) for running these tests locally

All test logs and artifacts are automatically uploaded and retained for 7 days for debugging purposes.

> **💡 Tip**: Run tests locally before pushing to catch issues early! See the [Testing section](#testing) below for quick start guides.

### Publish Artifacts (`.github/workflows/publish-artifacts.yml`)

**Artifacts published:**
- **Python Wheel**: Published to NVIDIA Artifactory
- **RAG Server Container**: Published to NGC Container Registry
- **Ingestor Server Container**: Published to NGC Container Registry
- **RAG Frontend Container**: Published to NGC Container Registry

All containers are tagged with both version-specific tags and `latest`.


## Best Practices

Follow these best practices when contributing:

### Branch Management
- Always create a feature branch from `develop` in your fork
- Use descriptive branch names (e.g., `feature/add-new-embedding-model`, `bugfix/fix-memory-leak`)
- Keep your fork's `develop` branch in sync with upstream
- Delete your feature branch after the PR is merged

### Commit Guidelines
- Write clear, concise commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers in commits when applicable (e.g., "Fix #123")
- Keep commits focused on a single concern
- Always sign-off your commits using `git commit -s`

### Code Standards
- Follow PEP 8 style guidelines for Python code
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality
- Ensure all tests pass locally before pushing

### Pull Request Guidelines
- Provide a clear description of the changes
- Link to related issues using GitHub keywords (Fixes #123, Closes #456)
- Ensure all CI checks pass before requesting review
- Respond to review comments promptly
- Keep PRs focused and reasonably sized (avoid massive PRs)
- Update documentation if your changes affect user-facing functionality

### Testing

Before pushing your PR, it's **strongly recommended** to run tests locally to catch issues early:

#### Running Unit Tests Locally
For detailed instructions on setting up and running unit tests locally, see [Unit Test Documentation](tests/unit/ci-unittest.md).

**Quick start:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -e .[all]
pip install -r tests/unit/requirements-test.txt

# Run unit tests with coverage
pytest tests/unit -v --cov=src --cov-report=term-missing
```

#### Running Integration Tests Locally
For comprehensive integration test setup and usage, see [Integration Test Documentation](tests/integration/README.md).

**Quick start:**
```bash
# Ensure services are running (RAG Server, Ingestor Server, Milvus)
# Install test dependencies
pip install -r tests/integration/requirements.txt

# Run basic integration tests
python -m tests.integration.main --rag-server http://localhost:8081 --ingestor-server http://localhost:8082

# List available test sequences
python -m tests.integration.main --list-sequences
```

#### General Testing Guidelines
- Run the full test suite locally before submitting
- Add unit tests for new functions and classes
- Add integration tests for new features when appropriate
- Verify your changes don't break existing functionality
- Test edge cases and error conditions
- Ensure all tests pass before pushing your changes

### Documentation
- Update README.md if adding new features
- Document configuration changes
- Add inline comments for complex logic
- Keep documentation in sync with code changes


## Signing Your Work
We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.
To sign off on a commit, use the `--signoff` (or `-s`) option when committing your changes:

`$ git commit -s -m "Add cool feature."`
This will append the following to your commit message:

Signed-off-by: Your Name <your@email.com>


## Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
