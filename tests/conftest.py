import os

# Tests never call paid APIs, whatever .env says.
os.environ["LLM_PROVIDER"] = "mock"
os.environ["JUDGE_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.data.generate import build  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def fresh_db():
    """Rebuild the deterministic database so tests never run against stale data."""
    build()
