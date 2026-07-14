"""Intent detection sub-awareness layer.

Turns a free-text description and/or a URL into structured ``IntentSignals``
that condition every generation modality with near-zero latency.  The signals
are serialised as ``[INTENT]`` awareness lines that outrank ``[HIGH]`` chart
signals and ``TRENDS:`` lines, giving user intent the highest priority in the
cascade.

Usage::

    from ai_model.intent import detect_intent

    signals = detect_intent(
        description="dark trap banger, neon-lit, very aggressive, 140 bpm",
        url="https://open.spotify.com/track/abc123",
    )
    # → IntentSignals(genre='trap', mood='dark', energy=0.88, ...)
    for line in signals.to_awareness_lines():
        print(line)
    # [INTENT] genre=trap mood=dark energy=0.88 tone=aggressive bpm=140 confidence=0.94
    # [INTENT] lighting=neon color_temp=cool
    # ...
"""
from .detector import IntentSignals, detect_intent
from .url_reader import UrlContent, read_url

__all__ = ["IntentSignals", "detect_intent", "UrlContent", "read_url"]
