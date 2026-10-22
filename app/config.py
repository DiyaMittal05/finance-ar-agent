"""Runtime configuration, read from environment variables."""
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    prompt_version: str = field(default_factory=lambda: os.getenv("PROMPT_VERSION", "v3"))
    judge_provider: str = field(default_factory=lambda: os.getenv("JUDGE_PROVIDER", "mock"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", ""))
    as_of_date: date = field(
        default_factory=lambda: date.fromisoformat(os.getenv("AS_OF_DATE", "2026-09-30"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", str(ROOT / "data" / "erp.db")))
    )
    docs_dir: Path = ROOT / "app" / "data" / "docs"
    # When set, the research agent calls a remote ERP service over HTTP.
    # When empty, it calls the same FastAPI app in-process.
    erp_base_url: str = field(default_factory=lambda: os.getenv("ERP_BASE_URL", ""))


settings = Settings()
