"""
Content Playbook — research-distilled world knowledge about top social content.

A curated, deterministic pattern library distilled from an extensive July-2026
web research pass over published engagement studies and creator-economy data
(Buffer 45M+ posts, Socialinsider/Later/Sprout multi-million-post caption
studies, Paddy Galloway's 3.3B-Shorts retention analysis, Later.com's 10k viral
TikTok hook study, Meta for Business caption-structure research, SSRN/PMC
emotional-arousal sharing research, and 2025-26 music-marketing playbooks).

What the research converged on:

  * Structure   — Hook -> Value -> CTA captions earn ~23% more engagement;
                  only the first ~125 characters are visible pre-fold, so the
                  hook must land there. 150-220 words is the IG comment-rate
                  sweet spot for long captions; Reels captions stay short.
  * Hooks       — the highest-retention opening archetypes are identity calls
                  ("if you're X..."), curiosity gaps, result previews, reveals,
                  and emotion-first storytelling. Hooks that resolve in under
                  2s beat 4-5s wind-ups by ~23% completion.
  * Emotion     — arousal level beats emotion type: awe, excitement and joy
                  are the most shareable non-divisive triggers.
  * CTAs        — low-friction beats generic: one-tap reactions ("drop a 🔥"),
                  tag-a-friend share triggers, and save prompts out-engage
                  plain "check it out" closers.
  * Hashtags    — 2025-26 platforms moved to quality-over-quantity; Instagram
                  hard-caps at 5, and 3-5 relevant tags is the working norm.
  * Music       — emotion-first context ("made this for the nights that...")
                  outperforms bare song clips; behind-the-scenes content is
                  the highest trust-builder for artists.

Like the quality buffer, this is borrowed world knowledge: the blending layer
(ai_model/quality_awareness.py) scales it by the same self-sufficiency weight,
so it retires as MaxBooster's own corpus grows. Everything here is pure data +
pure functions — no I/O, no model calls, never-raise.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

PLAYBOOK_VERSION = "2026-07-b"

# ── Hook archetypes (ranked by 2026 retention-test performance) ──────────────
# All templates are finished, user-facing copy with {idea}/{artist} slots.
# Every template targets ≥1 power word, ends with punctuation, emoji-ready.

HOOK_ARCHETYPES: Dict[str, List[str]] = {
    "identity_call": [
        "If {idea} is exactly what you've been waiting for — this one's yours 🎧",
        "This is for everyone who plays a song 40 times in a row: {idea} 🔁",
        "If you know, you know. {idea} just dropped and nothing is the same 🔥",
        "For the ones who find music before it's everywhere — {idea} is finally here 🎵",
        "You've been looking for this. {idea} is the record nobody warned you about 🔥",
        "If late nights and real emotions are your thing — {idea} was made for you 🌙",
        "Everyone who's ever felt too much is going to loop {idea} until sunrise 🌅",
        "This is for the ones who don't skip. {idea} earns every second 🎶",
        "Real ones already know. {idea} is the drop that separates the listeners 🔥",
        "If you've ever needed music to say what you couldn't — {idea} says it! 💯",
    ],
    "curiosity_gap": [
        "Here's what nobody tells you about {idea} — and it changes everything! 🔥",
        "There's one moment in {idea} you'll rewind twice. Find it first 👀",
        "The story behind {idea} is wilder than the song itself! 🤯",
        "Nobody is talking about what {idea} actually sounds like yet — but they will 🔥",
        "There's a reason {artist} kept {idea} quiet for this long — you'll hear it immediately 🎧",
        "Everyone is about to discover {idea}. You can be first 👀",
        "What happens when you actually sit with {idea} start to finish? Something shifts! 🎵",
        "The drop in {idea} that nobody saw coming just ended the conversation! 🔥",
        "There's a line in {idea} that's going to hit differently at 2am — trust! 🌙",
        "This is what {idea} sounds like when you play it at full volume — insane! 🔊",
    ],
    "result_preview": [
        "This is how {idea} sounds when it's finished — now watch how it started! 🎬",
        "The final version of {idea} first. The story behind it, next 🔥",
        "Played {idea} back and couldn't believe it was real — you need to hear this! 🎧",
        "{artist} made {idea} in one session. You can hear every second of it 🎵",
        "The room went silent when {idea} played back the first time — electric! ⚡",
        "After months of sessions — {idea} is finally what it was always supposed to be 🔥",
        "This is {idea} at its purest. No edits, no second guesses — just fire! 🔥",
        "Everything {artist} learned went into {idea}. The result is unreleased energy! ⚡",
        "Six months of work, one record: {idea}. Worth every second of the wait 🎶",
        "This is what happens when {artist} stops holding back — {idea} is the proof! 💥",
    ],
    "reveal": [
        "{artist} kept {idea} quiet for months. Not anymore 🔥",
        "What {artist} was really making this whole time: {idea} — drop everything and listen! 🎧",
        "Finally. {idea} is out and the wait is finally over — fire from the first bar! 🔥",
        "{artist} just revealed what {idea} actually sounds like. Nobody was ready 🤯",
        "The unreleased {idea} is finally here — and it's exactly what you thought it would be 🎵",
        "Exclusive: {idea} just surfaced and it's already changing the conversation! 💥",
        "Everything {artist} has been working toward finally arrives: {idea} 🔥",
        "The drop nobody saw coming is finally here — {idea} is out now! 🎶",
        "{artist} held {idea} back because it wasn't ready. Now it is — and it's fire! 🔥",
        "The secret {artist} kept for six months just dropped: {idea} — go listen now! 🎧",
    ],
    "emotion_story": [
        "{artist} made {idea} for the nights that don't make the highlight reel 🌙",
        "Some songs are written. {idea} was survived — you'll hear it in every bar! 🎵",
        "{idea} started as a voice memo at 3am. It became something nobody expected 🌙",
        "There's no performance in {idea}. Just {artist} being completely, finally honest 💯",
        "Every emotion {artist} couldn't say out loud for two years is in {idea} 🎶",
        "{idea} is the record {artist} needed to make before anything else made sense 🔥",
        "Not every song can be explained. {idea} is one of those — just press play 🎧",
        "{artist} wrote {idea} on the worst night of the year. It became the best record 💥",
        "The feeling {idea} carries is something {artist} has never put into words before 🎵",
        "Real, raw, and completely unfiltered — {idea} is {artist} at their most human 💯",
    ],
    "pattern_interrupt": [
        "Don't scroll — {idea} earns the next 15 seconds! 🔥",
        "Wait. Play {idea} out loud right now 🎧",
        "Stop. Read this first. {idea} just dropped and it's already viral 🔥",
        "You weren't ready for {idea} — but here it is anyway! 💥",
        "Pause whatever you're doing. {idea} is the drop you've been waiting for! 🎵",
        "Never heard anything like {idea} before — and that's not a comparison, it's a fact! 🔥",
        "This isn't a typical drop. {idea} is something completely different — fire! ⚡",
        "Stop sleeping on {artist} — {idea} just changed what's possible! 🎶",
        "Everyone who heard {idea} said the same thing first: this is insane! 🤯",
        "The moment you press play on {idea}, you'll understand why everyone's talking! 🔥",
    ],
    "social_proof": [
        "Everyone who's heard {idea} keeps sending it to people — find out why! 🎧",
        "{idea} is the record that people are texting each other at midnight! 🌙",
        "First day. {idea} is already on repeat in every city — insane rollout! 🔥",
        "The playlists are already adding {idea} — get there before everyone else! 🎵",
        "Everyone who finds {idea} early becomes an instant believer — join them! 🔥",
        "Real listeners know {idea} is special before the algorithm catches up! 💯",
        "The DMs about {idea} are flooding in — here's what they're all saying! 💬",
        "{idea} is spreading because it's genuinely different — listen and you'll know why! 🎶",
        "3 plays in and {idea} already lives in the 'send to everyone' folder! 🔁",
        "Nobody planned for {idea} to hit this hard — but here we are! 🔥",
    ],
    "behind_the_scenes": [
        "What nobody saw in the studio when {idea} was being made — exclusive! 🎬",
        "{artist} recorded {idea} at 4am. The mic caught something real! 🎙️",
        "The take that became {idea} wasn't planned — here's what actually happened! 🔥",
        "Three sessions in, {artist} scrapped everything and started {idea} from scratch! 💥",
        "The engineer on {idea} said it was the best session they'd ever recorded — fire! 🎧",
        "What the studio looked like when {idea} came together — pure creative energy! ⚡",
        "{artist} almost cut the bridge in {idea}. Kept it. You'll hear exactly why! 🎵",
        "The moment {idea} stopped being a work in progress and became something real! 🔥",
        "Nobody in the room believed {idea} would sound like this — and then it did! 🎶",
        "The production choice in {idea} that {artist} almost didn't make — exclusive! 🎬",
    ],
    "urgency": [
        "First day for {idea} — be in the first wave before the world catches on! 🔥",
        "Limited window to be early on {idea} — stream it now before it's everywhere! 🎧",
        "{idea} is out now and the first 24 hours set the trajectory — listen and share! 🔥",
        "The early supporters of {idea} are going to look very smart — join them now! 💯",
        "Stream {idea} today — the algorithm rewards early listeners and so does {artist}! 🎵",
        "Be the person who had {idea} on repeat before everyone else did — it's out now! 🔥",
        "Every stream of {idea} in the first 48 hours counts — don't sleep on this drop! ⚡",
        "Now or never: {idea} is live and the conversation is already starting — jump in! 🔥",
        "First plays matter. {idea} is out. You know what to do 🎶",
        "The window to be early on {idea} is right now — exclusive drop, go listen! 🎧",
    ],
}

# ── Genre-conditioned hooks ──────────────────────────────────────────────────
# Distinct cultural voice per genre — authentic to each community's language.

GENRE_HOOKS: Dict[str, List[str]] = {
    "drill": [
        "{idea} is different breed — cold, calculated, and built for the trenches! 🔥",
        "No hype, no cap: {idea} just reset the bar for what drill sounds like! 💯",
        "{artist} went crazy on {idea} — no features needed, no softening! 🔥",
        "The streets already know about {idea} — time the rest of the world caught up! 🎶",
        "Pressure makes diamonds. {idea} is what happens when {artist} applies both! 💎",
        "Cold flows, colder production — {idea} is built for the ones who know! 🔥",
    ],
    "afrobeats": [
        "{idea} is for the culture — the riddim, the feeling, the movement! 🌍",
        "The energy in {idea} doesn't need translation — it hits every timezone! 🔥",
        "{artist} brought everything to {idea} — joy, rhythm, and something that makes you move! 🎶",
        "Afro energy, global reach — {idea} is the record the world needs right now! 🌍",
        "From the studio to the streets to every dancefloor: {idea} is finally out! 🔥",
        "This one is for the culture. {idea} carries every influence {artist} grew up on! 🎵",
    ],
    "lofi": [
        "{idea} is for the 3am crowd — quiet, honest, and impossible to skip! 🌙",
        "The kind of music you put on and forget to turn off: {idea} just dropped! 🎧",
        "Not everything needs to be loud. {idea} hits hardest in the quiet moments! 🌙",
        "{artist} made {idea} for the late nights when words aren't enough 🌙",
        "One headphone in, rest of the world out — {idea} is that kind of record! 🎧",
        "Some music is made for the moment. {idea} was made for all the moments after! 🎵",
    ],
    "pop": [
        "{idea} is the anthem nobody knew they needed until right now! 🔥",
        "This is what pure pop excellence sounds like — {idea} is already everywhere! 🎶",
        "{artist} made {idea} for the moments you want to last forever — pure fire! 🔥",
        "The hook in {idea} lives in your head rent-free from the first listen! 🎵",
        "{idea} is radio-ready, playlist-certified, and absolutely unstoppable! 🔥",
        "Every generation has its anthem. {idea} is this moment's — and it's insane! 💥",
    ],
    "rnb": [
        "{idea} is the feeling you can't quite name — {artist} found the words! 🎵",
        "Smooth, soulful, and completely real — {idea} is {artist} at full power! 🔥",
        "The emotion in {idea} hits differently at night — you'll know exactly when! 🌙",
        "{artist} put everything vulnerable about this year into {idea} — fire! 🔥",
        "Late nights, deep feelings, no skips — {idea} is the R&B record of the season! 🎶",
        "The production on {idea} wraps around you like a memory — absolutely stunning! ✨",
    ],
    "hiphop": [
        "{artist} said everything that needed to be said on {idea} — bars from start to finish! 🔥",
        "This is hip-hop when it's doing what only hip-hop can do — {idea} is proof! 💯",
        "Lyrical, intentional, and completely elite — {idea} is already legendary! 🔥",
        "{artist} came with a different energy on {idea} — the culture is going to feel this! 🎶",
        "For everyone who said the bars were dead — play {idea} and come back! 🔥",
        "The wordplay in {idea} hits different every time you play it — pure genius! 💎",
    ],
    "indie": [
        "{artist} made {idea} away from the algorithm and the result is something real! 🎵",
        "Found a new favorite: {idea} sounds like nothing else dropping right now! 🎶",
        "{idea} is the record that reminds you why you fell in love with music — beautiful! 🔥",
        "No features, no formulas — just {artist} making {idea} exactly the way it needed to be! 💯",
        "The ones who find {idea} early will be recommending it for years! 🎵",
        "{artist} built {idea} for the listeners who actually pay attention — and it shows! 🔥",
    ],
    "edm": [
        "{idea} is built for the moment when the drop hits and time stops! ⚡",
        "{artist} engineered {idea} to physically move you — the bass doesn't lie! 🔊",
        "The build in {idea} is insane — the drop is even better! ⚡",
        "This is dancefloor science: {idea} makes your body move before your brain catches up! 🔥",
        "{idea} is the set-closer everyone's been waiting for — and it just dropped! ⚡",
        "When {idea} drops at 2am the whole room becomes one thing — electric! 🔥",
    ],
    "dancehall": [
        "{idea} is riddim, vibes, and pure energy from start to finish! 🌴",
        "The skanks are already starting — {idea} just hit the speakers! 🔥",
        "{artist} brought the island energy to {idea} and the whole world is moving! 🌍",
        "Dancehall don't lie: {idea} is the riddim everyone needed this season! 🌴",
        "From Kingston to everywhere — {idea} travels because it's just that good! 🔥",
        "Pure culture, pure vibes — {idea} is {artist} at their most authentic! 💯",
    ],
    "trap": [
        "{idea} is the sound that was missing — {artist} delivered and then some! 🔥",
        "Hard production, harder bars — {idea} doesn't ask for your attention, it takes it! 💥",
        "{artist} cooked something different on {idea} — the trap game just shifted! 🔥",
        "The 808s in {idea} are going to rearrange something in your chest — go listen! 🔊",
        "No skips, all heat — {idea} is {artist} at maximum pressure! 🔥",
        "The energy in {idea} is different from anything else dropping — it's insane! ⚡",
    ],
    "soul": [
        "{artist} put a lifetime of feeling into {idea} — this is what soul music is! 🎵",
        "The voice in {idea} carries something ancient and completely alive! 🔥",
        "{idea} sounds like the kind of record people play at the most important moments! 🎶",
        "Real soul music makes you feel seen — {idea} does that from the first note! 🔥",
        "{artist} wrote {idea} for everyone who's carried something heavy — fire! 💯",
        "There's no production trick in {idea} that matters more than the performance — stunning! ✨",
    ],
    "country": [
        "{artist} wrote {idea} from the truest place — and every word lands! 🎵",
        "Some songs take you somewhere specific. {idea} is that kind of record! 🔥",
        "Honest, unfiltered, and completely real — {idea} is country at its finest! 🎶",
        "The storytelling in {idea} makes you feel like you've lived that moment before! 🔥",
        "{artist} took real life and made it into {idea} — listen and you'll understand! 🎵",
        "No pretense, all truth: {idea} is why this genre exists! 💯",
    ],
    "jazz": [
        "{artist} took {idea} somewhere unexpected and made it feel inevitable! 🎵",
        "The conversation between the instruments in {idea} is unlike anything this year! 🎶",
        "{idea} rewards every listen with something new — this is jazz at its best! 🔥",
        "Complex, beautiful, and completely alive — {idea} is a masterclass! ✨",
        "{artist} found the space in {idea} that makes the silence as powerful as the notes! 🎵",
        "The improvisation in {idea} sounds planned because {artist} has done the work! 🔥",
    ],
}

# ── Platform-native language hooks ──────────────────────────────────────────
# Each platform has its own content DNA — these feel native, not imported.

PLATFORM_NATIVE_HOOKS: Dict[str, List[str]] = {
    "tiktok": [
        "POV: you find {idea} before it blows up 🔥",
        "the song that's about to be everywhere: {idea} — you heard it here first! 🎵",
        "wait until the second drop in {idea} — absolutely insane! 🤯",
        "this sound: {idea} — your fyp is about to be taken over! 🔥",
        "not me finding {idea} at midnight and immediately sending it to everyone 🌙",
        "the {idea} era has officially started and I'm not okay about it! 🔥",
        "tell me you're obsessed with {idea} without telling me — I'll go first! 🎧",
        "if {idea} doesn't end up on your summer playlist we can't be friends! ☀️",
    ],
    "instagram": [
        "{idea} deserves more than a 30-second clip — stream the whole thing ✨",
        "saved this because {idea} is a different level — tap link in bio 🎧",
        "the aesthetic of {idea} is exactly right for right now — stunning! ✨",
        "everything {artist} put into {idea} shows in every detail — extraordinary! 🔥",
        "quiet drop, loud impact: {idea} is already changing what's possible 🎵",
        "the cover art says it all. the music says more: {idea} is finally here! 🔥",
        "{idea} belongs on every curated playlist that matters right now ✨",
    ],
    "youtube": [
        "everything you need to know about {idea} — and why it changes everything! 🔥",
        "I broke down {idea} and the production choices are genuinely insane! 🎬",
        "why {idea} is being called one of the best drops of the year — watch till the end! 👀",
        "the real story behind {idea} — {artist} goes deep on what it actually took! 🎙️",
        "first reaction to {idea}: I wasn't ready. watch what happens! 🤯",
        "the techniques {artist} used in {idea} that nobody is talking about yet! 🎬",
        "{idea} — full reaction and breakdown. this record deserves the attention! 🔥",
    ],
    "twitter": [
        "{idea} really said 'I'm not the one' and delivered everything! 🔥",
        "nobody is talking enough about {idea} — criminal oversight! 🎵",
        "{idea} is in my head and I have no complaints about it! 🔁",
        "stream {idea} or we can't be friends — link in bio! 🎧",
        "the {idea} discourse starts now. who's ready? 🔥",
        "hot take: {idea} is {artist}'s best work. not debating this! 💯",
        "{artist} with {idea} is just unfair at this point! 🔥",
    ],
    "linkedin": [
        "What {artist}'s creative process on {idea} teaches us about creative risk! 💡",
        "Three principles {artist} used to make {idea} that apply beyond music! 💼",
        "The strategy behind {idea}'s rollout — and what independent artists can learn! 📊",
        "How {artist} built {idea} without industry support — a case study! 🚀",
        "What {idea} tells us about where music marketing is heading in 2026! 📈",
        "The business model behind {idea}: how {artist} monetized authenticity! 💡",
    ],
}

# ── Body archetypes with emotional arc ──────────────────────────────────────
# Three-act structure: tension/context → deepening → resolution/revelation.
# All use {idea}/{artist}/{genre}/{audience} slots.

BODY_ARCHETYPES: Dict[str, List[str]] = {
    "emotional_journey": [
        "{artist} sat with this one longer than anything else.\n\nEvery version that didn't make it taught {artist} something the final version carries.\n\n{idea} isn't the product of one session — it's the result of everything before it.",
        "There's a version of {idea} that was safer, smoother, and easier to explain.\n\n{artist} scrapped it.\n\nWhat's left is rawer, realer, and completely impossible to ignore.",
        "The hardest part of {idea} wasn't making it — it was deciding to release it.\n\n{artist} put everything real in there, left nothing safe.\n\nThat's exactly what you're going to hear.",
    ],
    "cultural_moment": [
        "{idea} arrives at exactly the right time — not by accident.\n\n{artist} has been listening to what {genre} needs and delivering something the culture actually asked for.\n\nThis is what it sounds like when an artist and a moment align.",
        "Music hits different when it's made for the people, not the algorithm.\n\n{artist} built {idea} for {audience} — the ones who actually feel it.\n\nThe charts will catch up later. The community knows now.",
        "Every era of {genre} has records that define it.\n\n{idea} is doing that work in real time — setting the tone, raising the standard, and making room for what comes next.\n\nPlay it and you'll hear why.",
    ],
    "production_focus": [
        "The production on {idea} is doing something most records don't attempt.\n\nEvery sound was chosen for a reason, every space was deliberate.\n\nThis is what music sounds like when the sonic choices match the emotional intent.",
        "{artist} and the production team on {idea} built something layered — you'll hear new things every time you play it.\n\nThat's intentional.\n\nSome records reward attention. {idea} demands it.",
        "There's a quiet detail in {idea} that only appears if you're actually listening.\n\n{artist} buried it there for a reason.\n\nTurn it up, pay attention, and find it.",
    ],
    "artist_perspective": [
        "{artist} needed to make {idea} before anything else made sense.\n\nIt started as a way to process something that didn't have language yet.\n\nIt ended as the most honest thing {artist} has ever put out.",
        "Everything {artist} wanted to say but couldn't — not yet — is in {idea}.\n\nThis is the record that unlocks what comes next.\n\nThe chapter before {idea} is over. This is what the new one sounds like.",
        "Some artists make music about their life. {artist} makes music that IS the life — unedited, unpolished in the right places, and completely real.\n\n{idea} is the proof of that. Go listen.",
    ],
    "listener_experience": [
        "First time you play {idea}, you'll catch the hook and the production.\n\nSecond time, you'll hear what's underneath.\n\nThird time, you'll understand why it matters.",
        "There's a moment in {idea} — you'll know it when you hit it — where everything {artist} was building toward lands at once.\n\nDon't skip to it.\n\nLet the whole journey earn it.",
        "{idea} is a different experience at different volumes.\n\nQuiet, it's intimate.\n\nLoud, it's something else entirely. Try it both ways.",
    ],
    "genre_authority": [
        "If you've been waiting for someone to do {genre} the right way — {artist} just answered that call with {idea}.\n\nEvery influence, every hour of listening, every lesson learned from the greats is in this record.\n\nThis is what the genre sounds like when it's taken seriously.",
        "{idea} advances {genre} without abandoning what makes it special.\n\n{artist} didn't chase what's trending in the genre — {artist} identified where it's going and got there first.\n\nThat's leadership.",
        "Real {genre} fans are going to recognize what {artist} did on {idea} immediately.\n\nThe references are there, but so is something completely new.\n\nThat balance is the hardest thing to get right. {artist} got it right.",
    ],
}

# ── CTA bank by intent (low-friction, research-backed phrasing) ─────────────

CTA_BANK: Dict[str, List[str]] = {
    "drive_streams": [
        "Save this so you're early when {idea} takes off 🎧",
        "Add {idea} to the playlist — link in bio",
        "Play {idea} once. That's the whole ask 🎵",
        "Stream {idea} now — link in bio. Every play counts in the first 48 hours! 🔥",
        "First streams set the momentum — go listen now, link in bio 🎧",
        "Put {idea} on your playlist before it's on every editorial one 🎵",
        "The algorithm rewards early listeners — be one of them, link in bio! 🔥",
        "Hit play. Add it. Share it. That's all {idea} needs from you today 🎶",
    ],
    "drive_engagement": [
        "Drop a 🔥 if {idea} hits different",
        "Tag someone who needs {idea} on their playlist right now! 👇",
        "One word for {idea} — comments are open 👇",
        "Tell me what {idea} reminds you of — I read every comment! 💬",
        "Who are you playing {idea} with first? Drop their name below! 👇",
        "Rate {idea} out of 10 in the comments — real ones only! 🔥",
        "If {idea} is going in your rotation, drop a 🎵 below so I know!",
        "Found a new favorite? Save it and tag someone who needs to hear it! 🎧",
    ],
    "grow_followers": [
        "Follow — the {idea} story is just getting started 🔔",
        "Stay close: everything behind {idea} drops here first! 🎵",
        "Follow now and be first for every drop after {idea}! 🔥",
        "The next chapter after {idea} is already in progress — follow to hear it first! 🔔",
        "If {idea} is what you've been looking for — you're in the right place. Follow! 🎶",
        "Don't miss what comes after {idea}. Follow and I'll make sure you hear it first! 🔥",
        "Follow for the full {artist} journey — {idea} is just the beginning! 🎵",
        "Subscribe and hit the bell — {idea} is the first of many! 🔔",
    ],
    "drive_conversion": [
        "Link in bio before it's gone 🛒",
        "First listeners get first access — link in bio! 🔥",
        "Limited window. Stream, save, and get the merch before it's everywhere — link in bio! 🛒",
        "The presave link is live — secure your spot before the release! 🎶",
        "Tickets, merch, and early access — everything is at the link in bio! 🔥",
        "Claim your early access before the window closes — link in bio! ⏰",
        "The drop is exclusive and the window is short — link in bio now! 🔥",
        "Secure the bag: first 100 get the early access deal — link in bio! 💯",
    ],
    "build_awareness": [
        "Share this with someone who hears music differently 🎵",
        "Remember where you heard {idea} first 👀",
        "Be the person who was on {idea} before it was everywhere! 🔥",
        "Send this to the one person in your life who will immediately understand it! 💬",
        "Share if {idea} is exactly the music you've been waiting for! 🎶",
        "Tag the friend who always finds the good music before everyone else! 👇",
        "The earlier you share {idea}, the better the story gets — pass it on! 🔥",
        "One share can change everything for an independent artist — do it for {idea}! 🙏",
    ],
}

# ── Video scene phrase templates ─────────────────────────────────────────────

SCENE_TEMPLATES: Dict[str, List[str]] = {
    "hook": [
        "Here's what nobody tells you about {idea} 🔥",
        "{artist} kept {idea} quiet for months. Not anymore 🎵",
        "Some songs are written. {idea} was survived 💯",
        "Don't scroll — {idea} earns the next 15 seconds! 🔥",
        "Wait. Play {idea} out loud right now 🎧",
        "You weren't ready for {idea} — but here it is! 💥",
        "If {idea} is what you've been looking for — your search is over! 🔥",
        "The drop in {idea} that nobody saw coming — exclusive reveal! 🎬",
        "Stop. {idea} just dropped and everything changed! 🔥",
        "First time anyone hears {idea} — watch what happens! 🤯",
        "Everyone who finds {idea} early becomes an instant believer 🎵",
        "{artist} said never again. Then made {idea} — and nobody is complaining! 🔥",
    ],
    "body": [
        "{artist} made {idea} for the nights that don't make the highlight reel 🌙",
        "Every second of {idea} was a choice — here's the one that mattered 🎶",
        "The room went quiet when {idea} played back the first time — electric! ⚡",
        "Real music hits differently when you know the story behind it! 🎵",
        "This is what {genre} sounds like when it's taken seriously — {idea} is the proof! 🔥",
        "Not every record needs to be explained — {idea} speaks for itself! 💯",
        "The production on {idea} rewards every listen with something new! 🎶",
        "{artist} built {idea} for the people who actually pay attention — and it shows! 🔥",
        "Every bar in {idea} was written with a purpose — and you'll feel each one! 🎵",
        "The energy {artist} brought to {idea} is impossible to fake — this is real! 💥",
        "Some records make you feel seen. {idea} is one of those — stream it now! 🔥",
        "The session that produced {idea} was different from the start — you'll hear why! 🎧",
    ],
    "cta": [
        "Save this so you're early when {idea} takes off 🎧",
        "Tag someone who needs {idea} on their playlist! 👇",
        "Play {idea} once. That's the whole ask 🎵",
        "Stream {idea} — link in bio — and share it! 🔥",
        "Drop a 🔥 if {idea} hits the way you thought it would! 💬",
        "Follow for everything that comes after {idea} — the journey continues! 🔔",
        "Add {idea} to your playlist today — you'll thank yourself later! 🎶",
        "Be early on {idea} — stream now and share the link! 🔥",
        "First streams matter most — go play {idea} right now! 🎧",
        "Comment your first reaction to {idea} — I read every one! 💬",
    ],
}

# ── High-arousal emotion lexicon ─────────────────────────────────────────────

HIGH_AROUSAL_WORDS = frozenset({
    "unbelievable", "insane", "chills", "goosebumps", "wild", "unreal",
    "wait", "finally", "obsessed", "loud", "alive", "electric", "explosive",
    "heart", "cry", "survived", "quiet", "rewind", "loop", "repeat",
    "first", "never", "everything", "nothing", "everyone", "nobody",
    "impossible", "unstoppable", "legendary", "masterpiece", "extraordinary",
    "breathtaking", "devastating", "euphoric", "transcendent", "visceral",
    "raw", "real", "honest", "vulnerable", "authentic", "undeniable",
    "fire", "drop", "viral", "exclusive", "secret", "proven", "limited",
    "unreleased", "breakthrough", "elite", "immaculate", "flawless",
})

# ── Platform norms (2025-26 research corrections) ────────────────────────────

HOOK_VISIBLE_CHARS = 125
HASHTAG_CAPS: Dict[str, int] = {
    "tiktok": 4, "instagram": 5, "instagram_reels": 5,
    "youtube": 3, "youtube_shorts": 3, "twitter": 2,
    "facebook": 3, "linkedin": 3, "general": 4,
}

# ── Genre-to-hook-style mapping ──────────────────────────────────────────────
# Maps a genre string to the best performing hook style for that genre.

GENRE_HOOK_STYLE: Dict[str, str] = {
    "drill": "pattern_interrupt", "uk drill": "pattern_interrupt",
    "trap": "pattern_interrupt", "grime": "pattern_interrupt",
    "afrobeats": "emotion_story", "afropop": "emotion_story",
    "dancehall": "social_proof", "reggae": "emotion_story",
    "lofi": "identity_call", "lo-fi": "identity_call",
    "ambient": "identity_call", "chillwave": "identity_call",
    "pop": "urgency", "electropop": "urgency",
    "rnb": "emotion_story", "r&b": "emotion_story", "soul": "emotion_story",
    "neo-soul": "emotion_story", "gospel": "emotion_story",
    "hip-hop": "reveal", "hip hop": "reveal", "rap": "reveal",
    "indie": "behind_the_scenes", "indie rock": "behind_the_scenes",
    "folk": "behind_the_scenes", "singer-songwriter": "emotion_story",
    "edm": "urgency", "house": "urgency", "techno": "pattern_interrupt",
    "electronic": "urgency", "dance": "urgency",
    "jazz": "curiosity_gap", "blues": "emotion_story",
    "classical": "result_preview", "orchestral": "result_preview",
    "country": "emotion_story", "country pop": "urgency",
    "rock": "pattern_interrupt", "alt rock": "pattern_interrupt",
    "metal": "pattern_interrupt", "punk": "pattern_interrupt",
}

# ── Research-backed creative directives ──────────────────────────────────────

_DIRECTIVES_CORE = [
    "Land the full hook inside the first 125 characters — that is all "
    "viewers see before the fold",
    "Aim for high-arousal emotion (awe, excitement, joy) — arousal level "
    "drives shares more than emotion type",
    "Structure as Hook → Value → CTA — this three-part shape earns 23% more engagement",
    "Use identity language ('this is for the ones who...') to create belonging before the ask",
]

_DIRECTIVES_BY_INTENT: Dict[str, str] = {
    "drive_streams": "Give the song context, not just a clip — emotion-first "
                     "framing converts viewers into listeners",
    "drive_engagement": "Close with a one-tap ask (emoji reaction or "
                        "tag-a-friend) — low friction wins comments",
    "grow_followers": "Tease a continuing story so following feels like "
                      "subscribing to the next chapter",
    "drive_conversion": "Pair the ask with urgency or early-access framing",
    "build_awareness": "Optimise for shareability — make the viewer look "
                       "good for passing it on",
}


# ── pure functions (all deterministic, never-raise) ──────────────────────────

def _normalise_genre(genre: Optional[str]) -> str:
    return (genre or "").lower().strip().replace("-", " ")


def genre_hook_candidates(topic: str, artist: str, genre: Optional[str]) -> List[str]:
    """Genre-conditioned hook candidates for the intelligence layer's ranking."""
    g = _normalise_genre(genre)
    bank: List[str] = []
    # Exact match first, then partial match
    pools = GENRE_HOOKS.get(g) or next(
        (v for k, v in GENRE_HOOKS.items() if k in g or g in k), []
    )
    idea = (topic or "").strip() or "this drop"
    art = (artist or "").strip() or "the artist"
    for tpl in pools:
        try:
            bank.append(tpl.format(idea=idea, artist=art))
        except (KeyError, IndexError, ValueError):
            continue
    return bank


def platform_hook_candidates(topic: str, artist: str, platform: str) -> List[str]:
    """Platform-native hook candidates."""
    plat = (platform or "").lower().strip()
    pool = PLATFORM_NATIVE_HOOKS.get(plat, [])
    idea = (topic or "").strip() or "this drop"
    art = (artist or "").strip() or "the artist"
    out: List[str] = []
    for tpl in pool:
        try:
            out.append(tpl.format(idea=idea, artist=art))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def hook_candidates(topic: str, artist: str) -> List[str]:
    """Formatted hook candidates from every researched archetype."""
    idea = (topic or "").strip() or "this drop"
    art = (artist or "").strip() or "the artist"
    out: List[str] = []
    for bank in HOOK_ARCHETYPES.values():
        for tpl in bank:
            try:
                out.append(tpl.format(idea=idea, artist=art))
            except (KeyError, IndexError, ValueError):
                continue
    return out


def body_candidates(
    topic: str, artist: str, genre: Optional[str] = None, audience: str = "listeners"
) -> List[str]:
    """Formatted body candidates with emotional arc."""
    idea = (topic or "").strip() or "this drop"
    art = (artist or "").strip() or "the artist"
    g = _normalise_genre(genre) or "music"
    out: List[str] = []
    for bank in BODY_ARCHETYPES.values():
        for tpl in bank:
            try:
                out.append(tpl.format(
                    idea=idea, artist=art, genre=g, audience=audience
                ))
            except (KeyError, IndexError, ValueError):
                continue
    return out


def cta_candidates(intent: str, topic: str) -> List[str]:
    """Formatted CTA candidates for an intent."""
    idea = (topic or "").strip() or "this"
    out: List[str] = []
    for tpl in CTA_BANK.get(intent, CTA_BANK["build_awareness"]):
        try:
            out.append(tpl.format(idea=idea))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def scene_phrase_templates(scene_type: str) -> List[str]:
    """Raw {idea}/{artist} phrase templates for a video scene pool."""
    key = {"hook": "hook", "drop": "hook", "build": "hook", "chorus": "hook",
           "cta": "cta", "outro": "cta"}.get(scene_type, "body")
    return list(SCENE_TEMPLATES.get(key, []))


def hashtag_cap(platform: str) -> int:
    return HASHTAG_CAPS.get(platform, HASHTAG_CAPS["general"])


def brief_directives(intent: str) -> List[str]:
    """Research-backed directives for the GenerationBrief (internal only)."""
    out = list(_DIRECTIVES_CORE)
    extra = _DIRECTIVES_BY_INTENT.get(intent)
    if extra:
        out.append(extra)
    return out


def best_genre_hook_style(genre: Optional[str]) -> Optional[str]:
    """Return the best-performing hook style for a given genre, or None."""
    g = _normalise_genre(genre)
    return GENRE_HOOK_STYLE.get(g) or next(
        (v for k, v in GENRE_HOOK_STYLE.items() if k in g or g in k), None
    )


_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def structure_score(text: str) -> float:
    """0-1 score for Hook->Value->CTA shape + arousal, per the research.

    Rewards: a front-loaded first line that fits inside the visible fold,
    a multi-part (hook/value/CTA) layout, high-arousal wording, and a
    low-friction interactive close. Deterministic and never-raise.
    """
    try:
        t = (text or "").strip()
        if not t:
            return 0.0
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        first = lines[0] if lines else t

        score = 0.0
        # Hook fits inside the pre-fold window (first 125 chars).
        if len(first) <= HOOK_VISIBLE_CHARS:
            score += 0.35
        # Distinct hook / value / CTA sections — 3+ lines is ideal structure.
        if len(lines) >= 3:
            score += 0.30
        elif len(lines) == 2:
            score += 0.15
        # High-arousal wording anywhere in the copy — weighted by density.
        low = t.lower()
        hits = sum(1 for w in HIGH_AROUSAL_WORDS if w in low)
        score += min(0.20, 0.05 * hits)
        # A low-friction interactive close (emoji ask / tag / save language).
        last = lines[-1].lower() if lines else low
        has_close_emoji = bool(_EMOJI_RE.search(lines[-1])) if lines else False
        has_close_cta = any(
            k in last for k in ("tag ", "save ", "drop a", "comment", "share", "follow", "stream", "link")
        )
        if has_close_emoji or has_close_cta:
            score += 0.15

        return min(1.0, round(score, 4))
    except Exception:  # noqa: BLE001 - scoring must never break generation
        return 0.0
