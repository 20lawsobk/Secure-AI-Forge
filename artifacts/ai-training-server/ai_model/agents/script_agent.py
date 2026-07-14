from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from ..model.creative_model import CreativeModel

# ── Static platform defaults — dead code in normal operation ─────────────────
# Kept as absolute last resort when both model and awareness are absent.

PLATFORM_HOOKS = {
    "tiktok":          "Stop scrolling — you need to hear this 🔥",
    "instagram":       "This is what you've been waiting for ✨",
    "youtube":         "In this video, I'm going to show you something incredible 🎬",
    "facebook":        "I've got something special to share today 🎵",
    "twitter":         "Thread time: the drop nobody saw coming 🔥",
    "linkedin":        "Here's a lesson from my music career that changes everything 💡",
    "google_business": "Exciting exclusive update from the studio 🎵",
    "threads":         "Let me tell you about the record that changed everything 🔥",
}

PLATFORM_CTAS = {
    "tiktok":          "Follow for more fire drops 🔥 — link in bio to stream now!",
    "instagram":       "Save this and tag someone who needs it 🎧 — follow for more drops!",
    "youtube":         "Subscribe and hit the bell 🔔 — never miss an exclusive drop!",
    "facebook":        "Share this with someone who needs to hear it 🎵 — follow the page!",
    "twitter":         "Repost if this goes hard 🔥 — stream link in bio!",
    "linkedin":        "Follow for exclusive music industry insights 💼 — share if this resonates!",
    "google_business": "Follow for live music updates 🎵 — stream and save the drop now!",
    "threads":         "Repost if this hits different 🔥 — drop a comment with your take!",
}


@dataclass
class ScriptRequest:
    idea: str
    platform: str
    goal: str
    tone: str
    awareness: str = ""
    variant_idx: int = 0
    genre: str = ""


@dataclass
class ScriptResponse:
    hook: str
    body: str
    cta: str
    source: str = "template"


def _clean_text(text: str) -> str:
    text = re.sub(r"<[A-Za-z_|][A-Za-z0-9_|/]*>", "", text)
    text = re.sub(r"\[(?:UNK|PAD|CLS|SEP|MASK|BOS|EOS)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        from ai_model.safety import enforce as _safety_enforce
        text = _safety_enforce(text)
    except Exception:
        pass
    return text


# ── Awareness parsing helpers ─────────────────────────────────────────────────

_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(always|never|make sure|ensure|emphasise|emphasize|include|avoid|"
    r"start with|open with|close with|end with|do not|don'?t|be sure)",
    re.IGNORECASE,
)


def _is_content_signal(text: str) -> bool:
    return not _INSTRUCTION_PREFIX_RE.match(text.strip())


def _any_lines(awareness: str, min_len: int = 15) -> List[str]:
    return [
        line.strip() for line in awareness.splitlines()
        if len(line.strip()) >= min_len and not line.strip().startswith("===")
    ]


def _parse_signals_for_platform(awareness: str, platform: str) -> List[str]:
    if not awareness:
        return []

    signals: List[str] = []
    plat_lower = platform.lower()
    other_platforms = [
        p for p in ["tiktok", "instagram", "youtube", "facebook", "twitter", "linkedin", "threads"]
        if p != plat_lower
    ]

    for line in awareness.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", stripped)
        if m:
            headline = m.group(2).strip()
            if plat_lower in headline.lower() or not any(p in headline.lower() for p in other_platforms):
                signals.append(headline)
        if stripped.startswith("Action:") or "↳ Action:" in stripped:
            action = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if action and len(action) > 15:
                signals.append(action)
        if stripped.startswith("•") and len(stripped) > 20:
            rec = stripped.lstrip("•").strip()
            if rec:
                signals.append(rec)
        # Plain caller-supplied awareness lines (no tier prefix) synchronize
        # WITH prefixed platform-buffer signals instead of being dropped once
        # any [HIGH] line exists. [INTENT] key=value lines are excluded — they
        # are machine-readable and must never leak into user-facing text.
        if not m and not stripped.startswith(("[", "•", "===", "Action:")) and "↳" not in stripped:
            plain = stripped[7:].strip() if stripped.startswith("TRENDS:") else stripped
            if len(plain) >= 20:
                signals.append(plain)

    if not signals:
        for line in awareness.splitlines():
            stripped = line.strip()
            m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", stripped)
            if m:
                signals.append(m.group(2).strip())
        if not signals:
            signals = _any_lines(awareness)

    return signals[:8]


# ── Expanded hook pools keyed by signal category ─────────────────────────────
# Each pool has 10+ entries to ensure genuine variety across variants.

_ALGORITHM_HOOKS: List[str] = [
    "The algorithm is finally pushing {idea} — drop everything and listen! 🔥",
    "This is what the viral algorithm wants right now: {idea}! 🔥",
    "The algorithm keeps surfacing {idea} — now you know exactly why! 🎯",
    "FYP is taking over with {idea} — and it's completely deserved! 🔥",
    "The algorithm found {idea} before the playlists did — you're early! 🎵",
    "When {idea} hits the FYP, you'll understand why the algorithm is right! 🔥",
    "Stop scrolling — {idea} is what the algorithm has been building toward! ⚡",
    "The recommendation engine finally knows what we knew: {idea} is different! 🎶",
    "Every feed is pushing {idea} for one reason — it's genuinely fire! 🔥",
    "Watch time is up, engagement is up — the algorithm found {idea} and so should you! 📈",
    "The viral loop starts now: {idea} is exactly what every algorithm rewards! 🔥",
    "From zero to everywhere: {idea} is what algorithmic justice looks like! 🎵",
]

_PLAYLIST_HOOKS: List[str] = [
    "Exclusive: playlist editors are watching — {idea} just landed! 🎧",
    "Editorial playlists are finally picking up {idea}! 🎧",
    "The playlist curators spotted {idea} — now it's your turn! 🎵",
    "Playlist placement confirmed: {idea} is officially in the ecosystem now! 🎶",
    "The editorial team just added {idea} — which means it's already too late to be first! 🔥",
    "Curators don't miss: {idea} just hit every list that matters! 🎧",
    "Discover Weekly, New Music Friday, mood playlists — {idea} is in all of them! 🎵",
    "The gatekeepers finally caught up: {idea} is on every editorial list now! 🔥",
    "Stream count is climbing because the playlists found {idea} first — join in! 🎧",
    "What the Spotify editorial team knows about {idea} that everyone else is about to learn! 🎵",
    "Playlist adds are accelerating for {idea} — be in the first wave of listeners! 🔥",
    "From hidden gem to editorial playlist: {idea} just made the jump — go stream! 🎧",
]

_SHORTFORM_HOOKS: List[str] = [
    "Short-form is dominating feeds — and {idea} is the drop everyone's waiting for! 🔥",
    "Reels are everywhere right now — {idea} is finally here for the moment! 🎬",
    "Vertical content is fire this week — {idea} was built for this exact moment! 🎬",
    "Short-form audio + {idea} is the combination that wins every algorithm! 🔥",
    "The Reels era just got its defining record: {idea} is fire from the first second! 🎵",
    "TikTok, Reels, Shorts — {idea} is dominating every vertical format right now! 🔥",
    "The 15-second hook culture finally met its match: {idea} delivers in every clip! ⚡",
    "Every creator is using {idea} right now — the reason is obvious once you hear it! 🔥",
    "Short-form proof: {idea} hits in any clip length, any format, any platform! 🎬",
    "The sound of short-form domination in 2026: {idea} — and it's not close! 🔥",
    "When every Reel and Short and TikTok is using {idea}, that's not coincidence! 🎵",
    "The new era of short-form content has a soundtrack: {idea} just announced it! 🔥",
]

_COLLAB_HOOKS: List[str] = [
    "The biggest viral trend on {platform} right now: {idea} — drop in now! 🔥",
    "Collabs are fire on {platform} — and {idea} just delivered something different! 🔥",
    "Everyone on {platform} is watching this drop — {idea} is finally here! 🎵",
    "The duet challenge for {idea} is already starting — be one of the first! 🔥",
    "Stitch this. Duet this. React to this. {idea} was built for every format! 🎬",
    "The collaborative energy around {idea} is unlike anything this week! 🔥",
    "When artists collaborate on something real, you get {idea} — pure fire! 🎵",
    "The co-sign culture found {idea} — which means the whole community is watching! 🔥",
    "Featured, remixed, or standalone: {idea} works every single way! 🎶",
    "The creative collab that produced {idea} is the conversation everyone's having! 🔥",
    "Collaboration goals: see {idea} for what it means when artists actually connect! 🎵",
    "Every major voice in the space is pointing at {idea} right now — for good reason! 🔥",
]

_EXCLUSIVE_HOOKS: List[str] = [
    "Exclusive and out now — {idea} is the drop that changes the conversation! 🔥",
    "First-ever listen to {idea} — and the reaction says everything! 🎧",
    "Limited window to be early on {idea} — stream it before it's everywhere! 🔥",
    "Exclusive release: {idea} is finally out and the timing is perfect! 🎵",
    "The exclusive drop that everyone's been waiting for: {idea} is here now! 🔥",
    "Only here first: {idea} before the world catches on — exclusive! 🎧",
    "Early access to {idea} is gone after today — go stream it right now! ⏰",
    "This is what exclusive sounds like: {idea} — first and only here! 🔥",
    "The exclusive stream is live: {idea} is out and it's already legendary! 🎵",
    "Limited, exclusive, and completely unrepeatable: {idea} is the drop of the season! 🔥",
    "The wait is finally over — {idea} exclusive drop is out now! 🎧",
    "Exclusive release: {idea} is the record that resets expectations! 🔥",
]

_TRENDING_HOOKS: List[str] = [
    "Trending everywhere for one reason: {idea} is genuinely different! 🔥",
    "The record that's been trending before it was even released: {idea} is here! 🎵",
    "Every conversation in music right now includes {idea} — for good reason! 🔥",
    "From trending topic to cultural moment: {idea} just made the jump! 🎶",
    "When something trends this fast, there's always a reason — {idea} is the proof! 🔥",
    "The trending sound that started with {idea} — and you're still early! 🎵",
    "Trending in every city, every genre, every playlist: {idea} is everywhere! 🔥",
    "The viral moment everyone's been waiting for: {idea} just landed! 💥",
    "From underground to trending in 48 hours: {idea} made history! 🔥",
    "Every dashboard shows {idea} trending — because it's genuinely that good! 📈",
    "Number one trending for a reason — {idea} hits different from the first play! 🔥",
    "Trending status confirmed: {idea} is the record that the whole platform is talking about! 🎵",
]

_VIRAL_HOOKS: List[str] = [
    "Viral for a reason: {idea} is the record that stops every scroll! 🔥",
    "The viral moment nobody planned for but everyone predicted: {idea}! 💥",
    "When something goes viral this fast, it's because it's real — play {idea}! 🔥",
    "The organic viral loop started with {idea} and it's nowhere near done! 🎵",
    "Zero paid promotion, pure virality: {idea} is doing it the old way! 🔥",
    "Every share of {idea} makes sense once you've heard it — go find out why! 🎶",
    "The most-shared release of the week: {idea} — and people aren't stopping! 🔥",
    "Viral before the algorithm: {idea} spread because people genuinely love it! 💯",
    "The share rate on {idea} is insane because the music is actually insane! 🔥",
    "When a record is this good, it makes itself viral: {idea} is living proof! 🎵",
    "From group chat to everywhere: {idea} went viral the right way — organically! 🔥",
    "Everyone who hears {idea} immediately sends it to someone — that's the definition of viral! 💥",
]

# Genre-conditioned body pools — authentic voice per genre
_GENRE_BODIES: Dict[str, List[str]] = {
    "drill": [
        "{idea} doesn't need hype — the production and the bars do all the talking.\n\nThis is cold, calculated, and made for the ones who understand what pressure sounds like.\n\nNo cap: this is different breed.",
        "Some records perform. {idea} just exists at a level others aspire to.\n\nThe delivery is immovable, the production is surgical.\n\nFor the ones who know, this is the confirmation they've been waiting for.",
        "{idea} is built for the trenches — every bar earned, nothing wasted.\n\nThis isn't music to impress the industry.\n\nThis is music for the community that knows what real sounds like.",
    ],
    "afrobeats": [
        "{idea} carries every influence {artist} grew up with — and delivers it somewhere new.\n\nThe riddim, the feeling, the movement: it's all there, and it's all real.\n\nFrom the studio to the streets to every dancefloor — this one travels.",
        "Global reach starts with authentic foundation.\n\n{idea} is built on real culture, real community, and a sound that transcends every border.\n\nThe world doesn't need to translate this — it just moves.",
        "When {artist} gets in the studio, the goal isn't charts — it's culture.\n\n{idea} is the result of that intention: infectious, genuine, and built to last.\n\nPlay it once and understand why afrobeats doesn't stop.",
    ],
    "lofi": [
        "{idea} was made for the moments between everything else — the quiet space where the real feelings live.\n\nNo performance, no production tricks: just something honest and perfectly imperfect.\n\nPut it on at 1am and understand why it exists.",
        "Some music fills silence. {idea} respects it.\n\nEvery note has space around it, every transition breathes.\n\nThis is what intentional sounds like when it's not trying to be anything other than real.",
        "{artist} made {idea} for the ones who listen, not just hear.\n\nThe details are in the quiet parts — the crackle, the space, the texture.\n\nThis is the kind of record that makes you stop what you're doing.",
    ],
    "pop": [
        "{idea} is engineered for the moment — the hook lands instantly, the production keeps you there.\n\nThis is pop that understands its job: make you feel something quickly and make it last.\n\nThe repeat button is going to work overtime.",
        "The best pop music feels inevitable after you hear it — like it was always supposed to exist.\n\n{idea} is that feeling from the first bar.\n\nCatch it now before it's in every commercial, every playlist, every moment.",
        "{idea} is built for everyone who needs their soundtrack right now.\n\nThe melody is undeniable, the production is flawless, the feeling is universal.\n\nThis is what pop sounds like when it's taken seriously.",
    ],
    "rnb": [
        "The emotion in {idea} doesn't announce itself — it arrives slowly and stays permanently.\n\n{artist} found the language for the feeling that usually has none.\n\nThis is what great R&B has always done: it makes the private feel universal.",
        "{idea} is the record for the late nights when you need something to articulate what you can't.\n\nSmooth where it needs to be, raw where it has to be.\n\nReal R&B for real moments.",
        "Some music tells you what to feel. {idea} helps you find what was already there.\n\n{artist} built this with patience, intention, and complete honesty.\n\nThis is R&B doing exactly what R&B was born to do.",
    ],
    "hiphop": [
        "{idea} is a full statement — bars, production, and intent aligned in a way that's rare.\n\nEvery line was placed with purpose, every beat chosen for a reason.\n\nThis is hip-hop when it's operating at its highest level.",
        "The lyricism in {idea} rewards attention.\n\nFirst listen you catch the concept. Second listen you hear the craft. Third listen you find what you missed.\n\nThat's what separates records from songs.",
        "{artist} came with a completely different energy on {idea}.\n\nThe confidence is earned, the delivery is immaculate, the production serves every bar.\n\nFor everyone who said the bars were dead — go listen to this record.",
    ],
    "indie": [
        "{artist} built {idea} away from the industry, away from the algorithm, and away from any commercial pressure.\n\nThe result is something that sounds like nothing else releasing right now.\n\nThat's the whole point — and it's exactly why it connects.",
        "Independent doesn't mean small. {idea} proves that the best music doesn't need the machine.\n\nIt's honest, it's detailed, and it trusts the listener to meet it where it lives.\n\nThe ones who find it first will never forget when they did.",
        "{idea} rewards the listeners who actually pay attention.\n\nEvery sound was chosen, every production decision was intentional.\n\nThis is what artistry looks like when commerce isn't the goal.",
    ],
    "edm": [
        "{idea} is built for the moment when everything drops away except the music.\n\nThe build is architectural — constructed layer by layer until the release feels inevitable.\n\nIn the right room, at the right time, this changes everything.",
        "Electronic music at its best creates physics — a physical response in the listener that bypasses thought.\n\n{idea} does that from the breakdown.\n\nThe engineering is precise, the emotion is overwhelming, and the drop is going to move you.",
        "{artist} engineered {idea} for the collective experience — the moment when a room of strangers becomes one thing.\n\nEvery frequency was tested, every transition was earned.\n\nThis is what happens when science meets feeling.",
    ],
}

# Tone-conditioned body pools — expanded for emotional depth
_TONE_BODIES: Dict[str, List[str]] = {
    "energetic": [
        "{idea} is exactly what the moment needs — and the energy is completely undeniable.\n\nThis is the drop that changes what's possible in {genre} right now.\n\nPlay it loud. You'll understand immediately.",
        "The energy in {idea} doesn't build — it arrives at full force and stays there.\n\nThis is {tone} music done the right way: committed from the first second to the last.\n\nDon't sleep on this drop.",
        "Everything is pointing toward {idea} right now — the timing, the sound, the energy.\n\nThis is the record for the ones who need something that matches their intensity.\n\nThe timing is perfect. Fire.",
        "{idea} is arriving at exactly the right moment — and the energy is impossible to ignore.\n\nHigh-tempo, high-impact, and built to dominate every playlist that matters.\n\nThis is what energetic actually means.",
        "If you need music that matches where you are right now — {idea} delivers without compromise.\n\nEvery second is intentional, every drop earned, every moment electric.\n\nThis is the drop you've been waiting for.",
    ],
    "professional": [
        "Presenting {idea} — crafted with intention and built for where the industry is going.\n\nEvery production decision was deliberate, every element serves the whole.\n\nThis is what professional sounds like when it's not just a descriptor.",
        "{idea} is built around what the industry is actually responding to right now — not what it responded to last year.\n\nForward-looking, meticulously crafted, and exclusively positioned.\n\nThis is serious music for serious listeners.",
        "{idea} is positioned for where the market is moving — and the positioning is precise.\n\nThe production is world-class, the execution is flawless, the timing is strategic.\n\nFinal release is live. The standard has been set.",
        "Industry-level production, artist-level authenticity: {idea} doesn't compromise on either.\n\nThis is what happens when craft and vision align without asking permission.\n\nExclusive. Now live. Make time for it.",
        "The benchmark just shifted. {idea} is what professional music looks like in 2026.\n\nEvery element was refined until nothing was wasted.\n\nThis is the release that defines this moment in the industry.",
    ],
    "casual": [
        "So {idea} just dropped — and yeah, it's finally that good.\n\nNot going to oversell it. Just going to say: go listen and come back.\n\nReal ones already know.",
        "Not gonna lie — {idea} is hitting completely different right now.\n\nI wasn't ready for it to be this good, and I don't think you are either.\n\nThat's the whole review. Go listen.",
        "Real talk: {idea} showed up and delivered everything and more.\n\nEveryone's talking about it, and for once the hype is completely justified.\n\nGo add it to the rotation — you'll thank yourself later.",
        "The {idea} drop hits different when you play it back-to-back.\n\nFirst time it's good. Second time it's better. Third time you understand why everyone's obsessed.\n\nThat's not a theory — go try it.",
        "Can't stop playing {idea} and I'm not even going to pretend I want to.\n\nThis is the one I've been sending to everyone.\n\nYou're next. Go listen.",
    ],
    "promotional": [
        "Introducing {idea} — available now on all platforms and already moving.\n\nStream it everywhere today and be part of the first wave.\n\nExclusive release, limited window to be early.",
        "{idea} is out now — stream it everywhere.\n\nThis is the drop that sets the tone for everything coming next.\n\nExclusive release. Limited first-wave window. Go now.",
        "{idea} just landed and the momentum is already building.\n\nGo listen now — first plays matter and every stream counts.\n\nThis is the drop you've been waiting for. It's finally here.",
        "Available everywhere right now: {idea} — the release that resets the standard.\n\nFirst 48 hours are the most important. Be in them.\n\nLink in bio. Stream it. Share it. Done.",
        "The launch of {idea} is live and the early response is exactly what we expected.\n\nBe one of the first listeners before the algorithm does it for you.\n\nStream it now. Tell someone. Repeat.",
    ],
    "authentic": [
        "{idea} is {artist} being completely honest — no performance, no filter, no compromise.\n\nThe kind of music you make when you stop trying to make music and just say something true.\n\nThis is that. Go listen and feel it.",
        "Real artists make real music. {idea} is the proof that the two things aren't always the same.\n\n{artist} built this from something genuine and left nothing safe behind.\n\nWhat you're about to hear is the real version.",
        "There's no version of {idea} that's more honest than this one.\n\n{artist} made every choice from an authentic place — the result is something that sounds unlike anything else.\n\nAuthentic doesn't mean raw. It means true. This is true.",
    ],
    "cinematic": [
        "{idea} is built like a film — scene by scene, emotion by emotion, building toward something inevitable.\n\nLet it play from the start. The journey is the point.\n\nThis is the cinematic experience that music can be when it's taken seriously.",
        "The production on {idea} paints something you can see.\n\nEvery sound serves the narrative, every transition advances the story.\n\nThis is what happens when a musician thinks like a director.",
        "{artist} treated {idea} like a film score and the result is transportive.\n\nCinematic, immersive, and impossible to listen to as background music.\n\nThis demands your full attention. It repays it completely.",
    ],
    "confident": [
        "{idea} is {artist} operating without doubt or apology.\n\nThe confidence isn't performed — it's built into every decision, every bar, every mix choice.\n\nThis is what artists sound like when they trust their instincts completely.",
        "No second guesses in {idea}. Every choice was the right one.\n\n{artist} knew exactly what this record needed to be and delivered it without compromise.\n\nConfidence sounds like this.",
        "{idea} is the statement of an artist who has nothing left to prove and chooses to prove it anyway.\n\nElite-level execution with zero hesitation.\n\nThis is what conviction sounds like in music form.",
    ],
    "playful": [
        "{idea} doesn't take itself too seriously — and that's exactly what makes it work.\n\nThe lightness is intentional, the fun is genuine, and the music underneath is seriously good.\n\nThis is what joy sounds like when it's not trying too hard.",
        "Sometimes music is supposed to make you smile first and think later.\n\n{idea} does both — in that order, perfectly.\n\nPress play and tell me the vibe isn't exactly right.",
        "{artist} made {idea} for the pure joy of making something that connects.\n\nPlayful, immediate, and impossible not to feel good about.\n\nThis is music that remembers what music is for.",
    ],
}

# Platform-native CTA pools
_PLATFORM_CTAS: Dict[str, List[str]] = {
    "tiktok": [
        "Follow for more fire drops 🔥 — link in bio to stream now!",
        "Drop a 🔥 in the comments if {idea} is going on the rotation!",
        "Stitch this and tell me your first reaction to {idea}! 🎬",
        "Duet this with your reaction — I want to see what {idea} does to everyone! 🔥",
        "Save this video before it disappears — {idea} link in bio! 🎵",
        "Comment one word for {idea} — I'm reading every single one 👇",
        "Tag someone in this who needs to hear {idea} before it's everywhere! 🔥",
        "Follow now — {idea} is just the beginning of what's coming! 🔔",
    ],
    "instagram": [
        "Save this and tag someone who needs it 🎧 — follow for more drops!",
        "Add {idea} to your playlist — link in bio 🎵",
        "Drop a comment with your first reaction 👇",
        "Share this to your story so your people hear {idea} too! 🔥",
        "Save this post — {idea} is the one you'll keep coming back to! 🎧",
        "Tag the friend who always finds the good music early 👇",
        "Link in bio — stream {idea} and come back and tell me! 🎵",
        "Follow the journey — everything behind {idea} drops here first 🔥",
    ],
    "youtube": [
        "Subscribe and hit the bell 🔔 — never miss an exclusive drop!",
        "Like this video and subscribe — more {idea} content coming! 🎬",
        "Comment your first reaction to {idea} below — I read every one! 💬",
        "Share this video if {idea} deserves more ears — it does! 🔥",
        "Subscribe for the full story behind {idea} dropping this week! 🔔",
        "Hit like and leave a comment — algorithm responds and so do I! 💬",
        "Turn on notifications — the next {idea} video drops this week! 🔔",
        "Share with someone who needs to discover {idea} — their reaction will be priceless! 🎬",
    ],
    "twitter": [
        "Repost if this goes hard 🔥 — stream link in bio!",
        "Stream {idea} and report back — I'll wait! 🎧",
        "Retweet this if {idea} deserves more plays than it's getting! 🔥",
        "Quote tweet with your rating — I'm reading every one! 💬",
        "Drop the link to your playlist if {idea} makes the cut! 🎵",
        "RT if you already knew this was going to be the one 🔥",
        "Reply with one word for {idea} — let's see what the consensus is! 💬",
        "Follow and you'll hear every drop before it becomes mainstream! 🔔",
    ],
    "linkedin": [
        "Follow for exclusive music industry insights 💼 — share if this resonates!",
        "Share this with someone in the music industry who should know about {idea}! 💼",
        "Follow for the business strategy behind {idea}'s rollout! 📊",
        "Connect and message me — happy to go deeper on the {idea} case study! 💡",
        "Share this post if you think independent artists deserve this level of visibility! 🔥",
        "Follow for weekly music industry analysis and exclusive behind-the-scenes! 📈",
    ],
    "facebook": [
        "Share this with someone who needs to hear it 🎵 — follow the page!",
        "Tag a friend who needs {idea} in their life right now! 👇",
        "Share this to your timeline — {idea} deserves every ear! 🔥",
        "Like and share if {idea} is exactly what you've been waiting for! 🎵",
        "Tag three people who would love {idea} — let's build the community! 👇",
        "Share this and tell me what platform you're streaming {idea} on! 🎧",
    ],
}


def _detect_genre_from_awareness(awareness: str, explicit_genre: str = "") -> str:
    """Detect genre from awareness string or explicit genre field."""
    if explicit_genre:
        return explicit_genre.lower().strip()
    if not awareness:
        return ""
    # Look for genre mentions in awareness
    genre_patterns = [
        "drill", "afrobeats", "lofi", "lo-fi", "pop", "rnb", "r&b",
        "hip-hop", "hip hop", "hiphop", "indie", "edm", "house", "techno",
        "electronic", "trap", "dancehall", "soul", "jazz", "country",
    ]
    aw_lower = awareness.lower()
    for g in genre_patterns:
        if g in aw_lower:
            return g
    return ""


def _parse_hook_from_awareness(
    awareness: str, platform: str, idea: str, signal_offset: int = 0
) -> str:
    """
    Build a data-informed hook sentence from live industry signals.
    Expanded pool of 80+ hooks across 7 signal categories.
    """
    if not awareness:
        return ""

    signals = _parse_signals_for_platform(awareness, platform)

    if signals:
        content_signals = [s for s in signals if _is_content_signal(s)]
        if not content_signals:
            content_signals = signals

        best = content_signals[signal_offset % len(content_signals)]
        best_lower = best.lower()

        if re.search(r"algorithm|watch time|engagement|trending|viral|fyp|foryou", best_lower):
            pool = _ALGORITHM_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"playlist|editorial|spotify|apple music|curator", best_lower):
            pool = _PLAYLIST_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"short.form|reels|shorts|vertical|tiktok|reel", best_lower):
            pool = _SHORTFORM_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"collab|feature|duet|stitch|trend|challenge", best_lower):
            pool = _COLLAB_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"exclusive|limited|first|early|unreleased", best_lower):
            pool = _EXCLUSIVE_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"trend|trending|top|chart|number one|rising", best_lower):
            pool = _TRENDING_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        if re.search(r"viral|spread|share|everywhere|exploding", best_lower):
            pool = _VIRAL_HOOKS
            return pool[signal_offset % len(pool)].format(idea=idea, platform=platform)

        # Direct signal as hook
        hook = best.split(":")[0].strip() if ":" in best else best
        hook = hook[:80].rstrip(",.")
        if len(hook) > 10:
            suffix = " 🔥" if not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", hook) else ""
            return f"{hook}!{suffix}" if not hook.endswith(("!", "?")) else f"{hook}{suffix}"

        _generic = [
            f"The industry is finally moving toward {idea} — and it's fire! 🔥",
            f"Now is the moment for {idea} — exclusive and ready to stream! 🎵",
            f"Everything is pointing to {idea} right now — drop everything and listen! 🔥",
            f"The timing for {idea} couldn't be more perfect — exclusive drop now! 🎧",
            f"The industry finally caught up: {idea} is exactly what's needed! 🔥",
        ]
        return _generic[signal_offset % len(_generic)]

    # awareness non-empty but no parsed signals
    trending_tags = re.findall(r"#(\w+)", awareness)
    if trending_tags:
        tag = trending_tags[signal_offset % len(trending_tags)]
        _synth = [
            f"#{tag} is finally everywhere — and {idea} is the drop you need! 🔥",
            f"#{tag} is viral this week — {idea} is the soundtrack! 🔥",
            f"Every feed is showing #{tag} — {idea} just dropped and it's fire! 🎵",
            f"#{tag} energy + {idea} = the combination everyone needed! 🔥",
            f"The #{tag} wave just found its record: {idea} is finally here! 🎶",
        ]
        return _synth[signal_offset % len(_synth)]

    _fallback = [
        f"The industry is finally here for {idea} — drop in now! 🔥",
        f"Now is the moment for {idea} — exclusive and ready! 🎧",
        f"Everything is pointing to {idea} right now — fire! 🔥",
        f"The timing couldn't be better: {idea} just landed! 🎵",
        f"First and exclusive: {idea} is out now — go listen! 🔥",
    ]
    return _fallback[signal_offset % len(_fallback)] if idea else "The moment is now! 🔥"


def _parse_cta_from_awareness(awareness: str, platform: str) -> str:
    if not awareness:
        return ""

    plat_lower = platform.lower()

    # 1. Platform-specific CTA line from awareness
    for line in awareness.splitlines():
        stripped = line.strip()
        if (len(stripped) <= 100
                and plat_lower in stripped.lower()
                and any(
                    kw in stripped.lower()
                    for kw in ["follow", "subscribe", "like", "share", "save",
                               "comment", "link", "stream"]
                )):
            cta = re.sub(r"^(•|↳|Action:)\s*", "", stripped).strip()
            if cta and len(cta) > 10:
                return cta[:120]

    # 2. Any action recommendation
    for line in awareness.splitlines():
        stripped = line.strip()
        if stripped.startswith("Action:") or "↳ Action:" in stripped:
            cta = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if cta and len(cta) > 10:
                return cta[:120]

    # 3. Use expanded platform-native CTA pool
    pool = _PLATFORM_CTAS.get(plat_lower, [])
    if pool:
        import hashlib
        # Use a simple deterministic selection based on awareness content
        idx = int(hashlib.md5(awareness[:50].encode()).hexdigest(), 16) % len(pool)
        return pool[idx]

    return PLATFORM_CTAS.get(plat_lower, "Follow for more exclusive drops 🔥 — stream now!")


def _build_awareness_body(
    awareness: str, platform: str, idea: str, tone: str,
    genre: str = "", signal_offset: int = 0
) -> str:
    """
    Build an emotionally resonant body from live industry context.
    Genre and tone conditioning produce distinct voices.
    """
    if not awareness:
        return ""

    signals = _parse_signals_for_platform(awareness, platform)
    trending_tags = re.findall(r"#(\w+)", awareness)
    topic_words = [t for t in trending_tags if len(t) > 4][:3]
    context_phrase = f" ({', '.join('#' + t for t in topic_words)} trending)" if topic_words else ""

    content_signals = [s for s in signals if _is_content_signal(s)]
    if not content_signals:
        content_signals = signals

    def _norm(s: str, maxlen: int = 60) -> str:
        cleaned = " ".join(s.split(":")[0].split()).rstrip(",.")
        if len(cleaned) <= maxlen:
            return cleaned
        truncated = cleaned[:maxlen]
        last_space = truncated.rfind(" ")
        return truncated[:last_space] if last_space > 0 else truncated

    # Genre-conditioned body takes priority over tone-conditioned
    genre_norm = _detect_genre_from_awareness(awareness, genre)
    genre_pool = _GENRE_BODIES.get(genre_norm, [])
    if genre_pool:
        return genre_pool[signal_offset % len(genre_pool)].format(
            idea=idea, artist="the artist", genre=genre_norm or "music", tone=tone
        )

    if len(content_signals) >= 2:
        i0 = signal_offset % len(content_signals)
        i1 = (signal_offset + 1) % len(content_signals)
        s1 = _norm(content_signals[i0])
        s2 = _norm(content_signals[i1])
        if i0 == i1:
            return f"{idea}{context_phrase}.\n\n{s1}."
        return f"{idea}{context_phrase}.\n\n{s1}.\n\n{s2}."
    elif content_signals:
        s1 = _norm(content_signals[0], maxlen=120)
        return f"{idea}{context_phrase}.\n\n{s1}."

    # Tone-conditioned pool — expanded
    if topic_words:
        tag_str = ", ".join("#" + t for t in topic_words)
        fallbacks = [
            f"{idea} — riding the viral wave of {tag_str} that's dominating right now.\n\nThe timing is finally here and {idea} delivers.",
            f"{idea} connects with the {tag_str} conversation happening across every feed.\n\nDrop in and be part of it — fire from the first bar.",
            f"{idea} is exactly where {tag_str} is pointing — and it's ready to drop now.\n\nEveryone who finds it early becomes a believer.",
            f"The {tag_str} energy finally found its record: {idea}.\n\nStream it everywhere and understand why the timing is perfect.",
        ]
        return fallbacks[signal_offset % len(fallbacks)]

    # Tone pools — expanded
    tone_pool = _TONE_BODIES.get(tone, _TONE_BODIES.get("energetic", []))
    if tone_pool:
        return tone_pool[signal_offset % len(tone_pool)].format(
            idea=idea, genre=genre_norm or "music", tone=tone, artist="the artist"
        )

    # Generic fallback with structure
    _generic = [
        f"{idea}{context_phrase} — shaped by what's working right now.\n\nFinally here, and it's fire from the first bar.",
        f"{idea}{context_phrase} — made for this exact moment.\n\nDrop everything and listen. The timing is perfect.",
        f"{idea}{context_phrase} — the timing couldn't be better.\n\nExclusive. Now live. Go stream it.",
        f"Real music for a real moment: {idea}{context_phrase}.\n\nNo hype needed — the music speaks for itself.",
    ]
    return _generic[signal_offset % len(_generic)]


# ── Agent ──────────────────────────────────────────────────────────────────────

class ScriptAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: ScriptRequest) -> ScriptResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        goal_token = f"<GOAL_{req.goal.upper()}>"
        tone_token = f"<TONE_{req.tone.upper()}>"

        awareness_prefix = ""
        if req.awareness:
            signals = _parse_signals_for_platform(req.awareness, req.platform)
            if signals:
                i0 = req.variant_idx % len(signals)
                i1 = (req.variant_idx + 1) % len(signals)
                awareness_prefix = f"Industry context: {signals[i0][:120]}\n"
                if len(signals) > 1 and i1 != i0:
                    awareness_prefix += f"Trend: {signals[i1][:80]}\n"
            trending_tags = re.findall(r"#(\w+)", req.awareness)
            if trending_tags:
                awareness_prefix += f"Trending: {', '.join(trending_tags[:4])}\n"

        prompt = f"{awareness_prefix}{platform_token} {goal_token} {tone_token} <STAGE_HOOK>"

        try:
            output = self.model.generate(prompt, max_new_tokens=80, temperature=0.8, top_p=0.92)

            hook = ""
            body = ""
            cta = ""

            if "<STAGE_BODY>" in output:
                hook_section = output.split("<STAGE_BODY>")[0]
                hook_section = hook_section.split("<STAGE_HOOK>")[-1] if "<STAGE_HOOK>" in hook_section else hook_section
                hook = _clean_text(hook_section)

                remainder = output.split("<STAGE_BODY>", 1)[1]
                if "<STAGE_CTA>" in remainder:
                    body = _clean_text(remainder.split("<STAGE_CTA>")[0])
                    cta = _clean_text(remainder.split("<STAGE_CTA>", 1)[1])
                else:
                    body = _clean_text(remainder)

            if not (self._is_meaningful(hook) and self._is_meaningful(body)):
                raw_lines = [
                    ln.strip() for ln in output.split("\n")
                    if ln.strip() and not (ln.strip().startswith("<") and ln.strip().endswith(">"))
                ]
                if len(raw_lines) >= 2:
                    hook = _clean_text(raw_lines[0])
                    body = _clean_text(" ".join(raw_lines[1:]))
                elif len(raw_lines) == 1 and len(raw_lines[0]) > 20:
                    sentences = re.split(r'(?<=[.!?])\s+', raw_lines[0], maxsplit=1)
                    if len(sentences) == 2 and len(sentences[0]) >= 10:
                        hook, body = _clean_text(sentences[0]), _clean_text(sentences[1])

            if self._is_meaningful(hook) and self._is_meaningful(body):
                from ai_model.request_intelligence import looks_garbled
                _wl = f"{req.idea} {req.awareness or ''}"
                if not looks_garbled(f"{hook}\n{body}", whitelist=_wl):
                    if not cta or not self._is_meaningful(cta) or looks_garbled(cta, whitelist=_wl):
                        cta = _parse_cta_from_awareness(req.awareness, req.platform)
                        if not cta:
                            cta = PLATFORM_CTAS.get(req.platform.lower(), "Let me know what you think!")
                    return ScriptResponse(hook=hook, body=body, cta=cta, source="ai_model")
        except Exception:
            pass

        return self._fallback(req)

    def _is_meaningful(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        control_count = sum(1 for w in words if w.startswith("<") and w.endswith(">"))
        if control_count / len(words) > 0.3:
            return False
        unique_words = set(w.lower() for w in words)
        if len(unique_words) < len(words) * 0.3:
            return False
        return True

    def _fallback(self, req: ScriptRequest) -> ScriptResponse:
        """
        Awareness-driven fallback with genre conditioning, expanded pools,
        and emotional arc structure. variant_idx ensures genuine variety.
        """
        platform = req.platform.lower().replace(" ", "_")
        awareness = req.awareness or ""
        genre = getattr(req, "genre", "") or ""

        if awareness:
            hook = _parse_hook_from_awareness(
                awareness, platform, req.idea, signal_offset=req.variant_idx
            )
            body = _build_awareness_body(
                awareness, platform, req.idea, req.tone,
                genre=genre, signal_offset=req.variant_idx
            )
            cta = _parse_cta_from_awareness(awareness, platform)
            return ScriptResponse(hook=hook, body=body, cta=cta, source="awareness")

        # True last resort — awareness absent
        hook = PLATFORM_HOOKS.get(platform, "Check this out 🔥")
        tone_pool = _TONE_BODIES.get(req.tone, [])
        if tone_pool:
            body = tone_pool[req.variant_idx % len(tone_pool)].format(
                idea=req.idea, genre=genre or "music", tone=req.tone, artist="the artist"
            )
        else:
            body = f"{req.idea} — made for this moment."
        cta = PLATFORM_CTAS.get(platform, "Let me know what you think!")
        return ScriptResponse(hook=hook, body=body, cta=cta, source="template")
