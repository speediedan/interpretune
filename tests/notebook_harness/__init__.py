"""Research-session harness for the test suite: drivers coupled to the fixture machinery in ``tests.conftest``.

Test infrastructure rather than example content, so it lives with the tests. The supported, test-free counterpart
that Hub experiment pipelines use is :mod:`interpretune.harness.sessions`.
"""

from tests.notebook_harness import session  # noqa: F401
