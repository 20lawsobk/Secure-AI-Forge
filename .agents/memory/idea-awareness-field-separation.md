---
name: Idea vs awareness field separation
description: Why raw topic/idea fields must never be concatenated with augmented request-intelligence context strings
---

Generation pipelines that template a raw `idea`/`topic` string directly into
user-facing output (e.g. `f"Stream {idea} now"`) must keep that field a
clean, short topic. Do not concatenate richer context — like a
request-intelligence `augmented_idea` (`"topic | tone: X | goal: Y | themes:
Z"`) — into the same field to "smuggle" extra signal into the pipeline.

**Why:** The concatenated pipe-separated string gets templated verbatim into
output text, then truncated by downstream length-trimming helpers, producing
garbled/cut-off user-facing text (e.g. `"Stream midnight drive | midnight
drive | tone:"`). This shipped silently because it still returned a
200/valid response — no error, just degraded quality.

**How to apply:** Give richer context its own dedicated field (e.g. an
`awareness` field, bullet-formatted so any awareness-signal parser picks it
up), never overload the same field that both (a) drives raw-string
templating and (b) is meant to carry structured steering context. When
reviewing/adding request-intelligence integration to a new generation
endpoint, grep every downstream usage of the field you're injecting into
before concatenating anything onto it.

**Caveat — don't just move the pollution into `awareness`.** `brief.directives`
(e.g. `"Optimise for: X"`, `"Open with a hook within the first line"`) are
internal prompt-engineering instructions for the model, not real-world
context. Some awareness parsers (e.g. script_agent's `_parse_signals_for_platform`)
naively treat *any* bulleted line as a quotable "signal" and echo it verbatim
into hook/body text. Only feed genuine external signals (real industry/trend
data, e.g. `req.awareness` from the client) into an `awareness` field that
feeds such a parser — never `brief.directives`. Check the specific parser's
extraction rules (does it require `[HIGH]/[MEDIUM]` tags, or does it also
match bare `•` bullets?) before deciding what's safe to put in `awareness`
for that agent. The video pipeline's scene-text sampler is a dedicated parser
that handles directives safely — script_agent's is not; don't assume the same
`awareness=directives` pattern is safe to copy across agents.
