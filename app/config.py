from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ClientCfg(BaseModel):
    id: str
    authorization: str
    user_id: str
    source_channel: str = "101"


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 3000
    api_key: str = ""


class Yun139Cfg(BaseModel):
    clients: list[ClientCfg]
    upstream_url: str = "https://ai.yun.139.com/api/outer/assistant/chat/v2/add"
    max_chars_per_request: int = 20000
    oversized_context_strategy: str = "truncate"
    idle_timeout_s: int = 15
    hard_timeout_s: int = 120
    # The web assistant will do a live web search before answering when
    # this is on, which adds real latency per request — often enough to
    # blow past Hermes's own client-side timeout on an otherwise-working
    # call. Off by default; turn on only if you specifically want
    # search-augmented answers and have widened Hermes's timeout to match.
    enable_network_search: bool = False
    # Makes the assistant rewrite the request into an image prompt and
    # actually call the Seedream text-to-image model when it judges an
    # image is wanted (observed via the mobile app's own capture — the
    # web capture has this off). When you turn this on, also raise
    # idle_timeout_s well above the default: there can be a ~50s+ gap
    # with zero bytes between the text finishing and the image result
    # arriving, which the default 15s idle watchdog would cut off early.
    enable_image_generation: bool = False
    # When enable_image_generation is on: if the latest user message looks
    # like an image request (matches image_intent_keywords), send it RAW —
    # skip the system-prompt/tool-manifest/history wrapping entirely —
    # instead of the normal fully-reconstructed dialogue. The wrapped
    # format (System:/User:/Assistant: labels, tool JSON schema prepended)
    # diverges enough from what the app itself sends that it can stop
    # yun139's own image-intent detection from firing. Normal turns are
    # completely unaffected by this.
    image_intent_bypass: bool = True
    image_intent_keywords: list[str] | None = None  # None = use the built-in default list


class ToolsCfg(BaseModel):
    enabled: bool = True


class ModelCfg(BaseModel):
    id: str = "yun139-lingxi"


class Config(BaseModel):
    server: ServerCfg
    yun139: Yun139Cfg
    tools: ToolsCfg = ToolsCfg()
    model: ModelCfg = ModelCfg()


def load_config() -> Config:
    path = Path(os.environ.get("CONFIG_PATH", "config/config.yaml"))
    if not path.exists():
        example = path.with_name(path.stem + ".example" + path.suffix)
        raise FileNotFoundError(
            f"Config not found at {path}. Copy {example} to {path} and fill in your credentials."
        )
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Config.model_validate(raw)


class ClientPool:
    """Simple round-robin over configured accounts (mirrors Gemini-FastAPI's
    multi-client list, minus health-tracking — add that if you actually run
    more than one account)."""

    def __init__(self, clients: list[ClientCfg]):
        if not clients:
            raise ValueError("yun139.clients must have at least one entry")
        self._cycle = itertools.cycle(clients)

    def next(self) -> ClientCfg:
        return next(self._cycle)
