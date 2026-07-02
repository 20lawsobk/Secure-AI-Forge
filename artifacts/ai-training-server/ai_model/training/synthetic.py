from __future__ import annotations
import json
import os
import random

PLATFORMS = ["TikTok", "Instagram", "YouTube", "Facebook", "Twitter", "LinkedIn"]
TONES = ["energetic", "chill", "inspirational", "edgy", "playful", "professional", "casual", "promotional"]
GENRES = ["hip-hop", "R&B", "pop", "trap", "soul", "electronic", "afrobeats", "drill", "lofi", "indie", "reggaeton", "jazz", "acoustic"]
GOALS = ["streams", "followers", "engagement", "sales", "awareness", "growth", "conversion"]
AD_HOOKS = [
    "POV: your music goes viral overnight",
    "Stop scrolling — you need to hear this",
    "Finally. A beat that hits different.",
    "The sound everyone's been waiting for",
    "This song just broke my algorithm",
    "I wasn't ready for this drop",
    "Your playlist needs this right now",
    "Listen to this before it gets big",
    "The track your feed has been missing",
    "This is what your morning commute needs",
    "Everyone is sleeping on this artist",
    "Turn your speakers up for this one",
]
CTA_LIST = ["Stream Now", "Listen Free", "Follow for More", "Drop in Comments", "Share if You Vibe",
            "Save This Track", "Add to Playlist", "Tap the Link", "Pre-Save Today", "Turn On Notifications"]
SOCIAL_HOOKS = [
    "Day {n} of dropping fire until I blow up",
    "No one talks about this part of being an indie artist",
    "How I went from 0 to {n}k streams in 30 days",
    "The beat I made at 3am just hit different",
    "Behind the scenes: recording '{title}'",
    "Real talk — the music industry doesn't want you to know this",
    "POV: you discover your new favorite artist",
    "My most personal song yet just dropped",
    "This almost didn't make the album",
    "What {n} months of studio work actually looks like",
    "The story behind '{title}' will surprise you",
    "I almost quit music before '{title}' blew up",
]
BODY_TEMPLATES = [
    "Been working on this for {months} months. Every late night, every early morning went into this. The {genre} vibes are real on this one. Link in bio to stream.",
    "This track came from a place of {tone} energy — {genre} at its core, built for the fans who get it. {months} months of writing, mixing, and second-guessing led to this.",
    "Straight from the studio to your speakers. {genre} inspired, {tone} in tone, and built to make you feel something. Give it a spin.",
    "{months} months, countless versions, one final cut. This is the {genre} record I needed to make. Let me know what you think.",
    "Made this one for the late-night drives and the early-morning grind. {genre} roots, {tone} delivery. Out now everywhere.",
]
TITLES = [
    "Midnight Sessions", "Neon Dreams", "Gold Rush", "Frequency", "Ascension",
    "Raw Footage", "Street Gospel", "Wavelength", "Overtime", "The Come Up",
    "Afterglow", "Static", "Lowlight", "Northbound", "Payphone", "Echo Chamber",
]
HASHTAG_POOL = ["NewMusic", "IndieArtist", "MusicProducer", "HipHop", "RnB", "NewArtist",
                "MusicLife", "StudioSession", "SupportIndieArtists", "NewRelease", "MusicMarketing"]
AUDIENCE_POOL = ["hip-hop fans", "music lovers", "playlist listeners", "R&B fans",
                 "urban culture", "trap enthusiasts", "indie music fans", "gen z listeners",
                 "college radio", "underground hip-hop heads", "afrobeats fans", "lofi listeners"]
LYRIC_HOOKS = [
    "I been on the grind since day one, nobody saw me coming",
    "We made it out, we ain't going back to nothing",
    "Every night I pray that I can find my way",
    "The city lights reflect in my eyes at midnight",
    "They never believed until the whole world was watching",
    "Running through the static just to find your signal",
    "Every scar's a story I ain't scared to tell",
    "We built this from nothing, brick by brick, night by night",
    "Chasing the sound that lives inside my head",
    "This ain't a comeback, I never left",
]
STRATEGIES = [
    "pitch to editorial playlists 7 days before release",
    "pre-save campaign starting 21 days out",
    "release day social blitz across all platforms",
    "TikTok pre-release audio leak strategy",
    "influencer seeding 10 days before release",
    "countdown story series across the final week",
    "exclusive early access for email subscribers",
    "coordinated short-form video drop across all platforms on release day",
]
VIDEO_SCRIPT_HOOKS = [
    "The hook happens in the first 2 seconds or you lose them",
    "This is how you build a beat drop that actually hits",
    "Watch the transition at 0:08 — that's the whole trick",
    "This video format got us 3x the normal engagement",
    "Here's the exact shot list we used for this video",
]
PLAYLIST_PITCH_TEMPLATES = [
    "Submitting '{title}' for playlist consideration — {genre} record with {tone} energy, perfect fit for editorial and algorithmic {genre} playlists.",
    "'{title}' is a {tone} {genre} track built for late-night and focus playlists — clean mix, radio-ready hook, {bpm} BPM.",
    "Pitching '{title}' to curators: {genre} sound, {tone} mood, designed to fit seamlessly into {genre} discovery playlists.",
]


def _random_social_post() -> dict:
    platform = random.choice(PLATFORMS)
    tone = random.choice(TONES)
    goal = random.choice(GOALS)
    genre = random.choice(GENRES)
    hook = random.choice(SOCIAL_HOOKS).format(
        n=random.randint(1, 100),
        title=random.choice(TITLES),
    )
    body = random.choice(BODY_TEMPLATES).format(
        months=random.randint(2, 18), genre=genre, tone=tone,
    )
    cta = random.choice(CTA_LIST)
    hashtags = " ".join([f"#{h}" for h in random.sample(HASHTAG_POOL, k=random.randint(3, 6))])
    return {
        "platform": platform,
        "tone": tone,
        "goal": goal,
        "genre": genre,
        "hook": hook,
        "body": body,
        "cta": cta,
        "hashtags": hashtags,
        "text": f"{hook}\n\n{body}\n\n{cta}\n\n{hashtags}",
    }


def _random_ad_creative() -> dict:
    platform = random.choice(["TikTok", "Meta", "YouTube"])
    tone = random.choice(TONES)
    hook = random.choice(AD_HOOKS)
    cta = random.choice(CTA_LIST)
    product = f"{random.choice(TITLES)} {random.choice(['EP', 'Single', 'Album', 'Tape'])}"
    goal = random.choice(GOALS)
    audience = random.sample(AUDIENCE_POOL, k=2)
    return {
        "platform": platform,
        "tone": tone,
        "hook": hook,
        "cta": cta,
        "product": product,
        "goal": goal,
        "audience": audience,
        "text": f"{hook} | {product} | {cta} | Target: {', '.join(audience)}",
    }


def _random_daw_content() -> dict:
    title = random.choice(TITLES)
    genre = random.choice(GENRES)
    tone = random.choice(TONES)
    goal = random.choice(GOALS)
    bpm = random.randint(70, 160)
    key = random.choice(["C minor", "G major", "F# minor", "Bb major", "D minor", "A major"])
    lyric_hook = random.choice(LYRIC_HOOKS)
    return {
        "title": title,
        "genre": genre,
        "tone": tone,
        "goal": goal,
        "bpm": bpm,
        "key": key,
        "lyric_hook": lyric_hook,
        "beat_description": f"{tone.capitalize()} {genre} beat, {bpm} BPM, {key}, layered 808s with melodic top line",
        "text": f"Track: {title} | Genre: {genre} | BPM: {bpm} | Key: {key} | Hook: {lyric_hook}",
    }


def _random_distribution_plan() -> dict:
    title = random.choice(TITLES)
    release_in = random.randint(14, 60)
    platforms = random.sample(["Spotify", "Apple Music", "Tidal", "Amazon Music", "YouTube Music", "Deezer"], k=4)
    strategy = random.choice(STRATEGIES)
    goal = random.choice(GOALS)
    return {
        "title": title,
        "goal": goal,
        "release_days_out": release_in,
        "platforms": platforms,
        "strategy": strategy,
        "text": (
            f"Release '{title}' in {release_in} days on {', '.join(platforms)}. "
            f"Strategy: {strategy}."
        ),
    }


def _random_video_script() -> dict:
    platform = random.choice(PLATFORMS)
    tone = random.choice(TONES)
    genre = random.choice(GENRES)
    goal = random.choice(GOALS)
    hook = random.choice(VIDEO_SCRIPT_HOOKS)
    title = random.choice(TITLES)
    scene = (
        f"Open on a close-up of the mixing board, cut to the artist recording '{title}', "
        f"drop the {genre} beat on the downbeat, end on a text overlay with the CTA."
    )
    cta = random.choice(CTA_LIST)
    return {
        "platform": platform,
        "tone": tone,
        "genre": genre,
        "goal": goal,
        "hook": hook,
        "scene": scene,
        "cta": cta,
        "text": f"{hook}. {scene} {cta}.",
    }


def _random_playlist_pitch() -> dict:
    title = random.choice(TITLES)
    genre = random.choice(GENRES)
    tone = random.choice(TONES)
    bpm = random.randint(70, 160)
    goal = random.choice(GOALS)
    text = random.choice(PLAYLIST_PITCH_TEMPLATES).format(title=title, genre=genre, tone=tone, bpm=bpm)
    return {
        "title": title,
        "genre": genre,
        "tone": tone,
        "goal": goal,
        "bpm": bpm,
        "text": text,
    }


def generate_synthetic_samples(path: str, n: int = 500):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    generators = [
        _random_social_post,
        _random_ad_creative,
        _random_daw_content,
        _random_distribution_plan,
        _random_video_script,
        _random_playlist_pitch,
    ]

    samples = []
    for i in range(n):
        gen = generators[i % len(generators)]
        samples.append(gen())

    random.shuffle(samples)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"[synthetic] Generated {n} samples → {path}")
    return samples
