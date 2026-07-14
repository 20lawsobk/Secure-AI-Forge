"""Universal URL Parser — public API.

Import :func:`parse_url` to get a rich :class:`ParsedUrl` for any URL.
The parser is never-raise: it always returns a :class:`ParsedUrl` even on
network failures or malformed input, with whatever partial data it could
extract.

Usage::

    from ai_model.url_parser import parse_url

    result = parse_url("https://open.spotify.com/track/6rqhFgbbKwnb9MLmUQDhG6")
    print(result.artist)      # "Drake"
    print(result.title)       # "Gods Plan"
    print(result.platform)    # "spotify"
    print(result.topic_string)  # "Gods Plan — Drake (Spotify track)"
    print(result.awareness_text)  # multi-line awareness block for ScriptAgent
"""
from __future__ import annotations

from .models import ParsedUrl
from .core import parse_url, parse_topic_url

__all__ = ["ParsedUrl", "parse_url", "parse_topic_url"]
