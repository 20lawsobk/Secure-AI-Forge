---
name: CTA leak guards
description: Rules for CTA selection from awareness text — coaching lines and template placeholders must never ship verbatim.
---

# CTA leak guards (script agent)

**Rule 1:** CTA candidates pulled from awareness text must skip coaching/playbook
lines — `[HIGH]/[MED]/[LOW]`-tagged research lines and imperative coaching
("Open with…", "Lead with…", "Action: Open with…"). Strip list/action markers
(`•`, `↳`, `Action:`) BEFORE testing openers, or "Action: Open with…" slips through.

**Rule 2:** Platform CTA template pools contain `{idea}` placeholders; any
selector reading from a pool must format (or fall back to a placeholder-free
entry) — an unformatted literal `{idea}` shipped to users once.

**Why:** The platform awareness buffer (quality_awareness playbook) contains
strategy notes ABOUT content, not copy FOR content; the awareness→CTA heuristic
picked them verbatim ("Open with an identity call or pattern interrupt…").

**How to apply:** Any new code path that mines awareness/playbook text for
user-facing copy needs both guards; the URL-topic tests in the wave-6 quality
harness assert them (garble + metadata-echo + `{idea}` placeholder checks).
