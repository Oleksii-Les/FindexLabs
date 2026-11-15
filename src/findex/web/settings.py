"""Configuration: every knob comes from the environment (twelve-factor), nothing from a file.

    INDEX_PATH=/data/index.bin   the index to serve (the one required setting in production)
    EMBEDDINGS_PATH=/data/emb    prefix of the files made by `findex embed` (optional: semantic and hybrid modes)
    CORPUS_ROOT=/data/books      where the .txt files live if the paths inside the index are stale
    PORT=8000                    the port (Render, Railway and Fly.io set it for you)
    WEB_CONCURRENCY=2            number of worker processes
    LOG_LEVEL=INFO  CACHE_SIZE=256  VECTORIZED=true  SNIPPET_WINDOW=110  SITE_NAME=findex  EXAMPLE_QUERIES="whale,white whale"

Unknown variables are ignored; a bad value (``PORT=abc``) stops the program at start-up with a
message naming the variable, instead of failing on the first request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    index_path: Path = Path("index.bin")
    corpus_root: Path | None = None
    embeddings_path: Path | None = None  # prefix made by `findex embed`; unset = no semantic modes
    port: int = Field(default=8000, ge=1, le=65535)
    web_concurrency: int = Field(default=1, ge=1, le=64)  # env WEB_CONCURRENCY: worker processes
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cache_size: int = Field(default=256, ge=0)
    vectorized: bool = (
        True  # NumPy scoring (lab 8); False = the pure-Python loop of lab 3, for comparisons
    )
    snippet_window: int = Field(default=110, ge=20, le=500)
    site_name: str = Field(default="findex", min_length=1, max_length=40)
    example_queries: str = "whale,white whale,ocean AND storm,love OR death"

    @property
    def workers(self) -> int:
        return self.web_concurrency

    @property
    def examples(self) -> list[str]:
        return [q.strip() for q in self.example_queries.split(",") if q.strip()]
