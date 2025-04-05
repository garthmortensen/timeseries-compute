# Architecture Decision Records

## ADR-001: Modular Library Design

### Context
- The package was originally part of a larger thesis replication project
- Need to balance functionality with maintainability
- Options:
  1. Maintain as part of the larger project
     - pros: unified codebase, simpler integration
     - cons: harder to maintain, test, and reuse independently
  2. Extract as a standalone package
     - pros: focused scope, increased reusability, better testing
     - cons: requires additional integration effort when used with other components

### Decision and Consequences
Decision: Extract as a standalone Python package.

Consequences: Increased maintainability and reusability at the cost of additional integration work. Enables independent versioning and easier contribution from community.

## ADR-002: Modern Build System

### Context
- Python offers multiple packaging approaches:
  1. Legacy `setup.py`
     - pros: widely understood, extensive documentation
     - cons: less structured, harder to manage dependencies
  2. `pyproject.toml` ([PEP 621](https://peps.python.org/pep-0621/))
     - pros: declarative, standardized, better dependency management. It's the *new* thing, and it's backed by a PEP.
     - cons: newer standard with less historical examples

### Decision and Consequences
Decision: Adopt `pyproject.toml` for packaging.

Consequences: More robust dependency management and simpler package definition. Better compatibility with modern Python tooling at the cost of potential friction with older tools.

## ADR-003: Documentation Strategy

### Context
- Documentation approaches:
  1. Separate documentation files
     - pros: centralized, easier to maintain
     - cons: can become outdated when code changes
  2. Self-documenting code via docstrings and type hints
     - pros: documentation stays close to code, automatic API doc generation
     - cons: clutters up the code, if excessive

### Decision and Consequences
Decision: Implement extensive docstrings and type hints with Sphinx documentation generation.

Consequences: Documentation remains close to code and is automatically generated, reducing the risk of outdated docs. Code is more explicit but requires discipline to maintain high-quality docstrings.

## ADR-004: Testing and Quality Assurance

### Context
- Testing strategies:
  1. Manual testing
     - pros: human judgment on edge cases
     - cons: time-consuming, prone to human error, not automated
  2. Automated testing with code coverage metrics
     - pros: consistent, repeatable, can be integrated into CI/CD
     - cons: requires additional development time, may not catch all edge cases

### Decision and Consequences
Decision: Implement comprehensive unit tests with pytest targeting at least 70% code coverage.

Consequences: Higher code quality and confidence when making changes. Additional development time is offset by reduced debugging time and easier maintenance.

## ADR-005: Cross-Platform and Multi-Version Compatibility

### Context
- Support requirements:
  1. Single platform/version focus
     - pros: simpler development, fewer edge cases
     - cons: limited user base
  2. Cross-platform and multi-version support
     - pros: broader user base, more flexible
     - cons: more complex testing matrix, slower processing (Windows and Mac!), additional maintenance, additional code complexity.

### Decision and Consequences
Decision: Support cross-platform (Windows, Linux, macOS) and multiple Python versions (3.11+).

Consequences: Broader user base at the cost of more complex testing. Automated testing across platforms and versions using GitHub Actions helps manage this complexity.

## ADR-006: Containerization Strategy

### Context
- Deployment options:
  1. Python package only
     - pros: simpler, lighter weight
     - cons: users must manage dependencies
  2. Containerized deployment
     - pros: consistent environment, easier deployment
     - cons: additional complexity, larger footprint

### Decision and Consequences
Decision: Provide Docker support with optimized multi-stage builds.

Consequences: Easier deployment and consistent environments for users at the cost of additional maintenance of Docker assets. Multi-stage builds help reduce the final image size.
