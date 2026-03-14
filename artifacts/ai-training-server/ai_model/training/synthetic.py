from __future__ import annotations
import json
import os
import random

PLATFORMS = ["TikTok", "Instagram", "YouTube", "Facebook", "Twitter", "LinkedIn"]
TONES = ["energetic", "chill", "inspirational", "edgy", "playful", "professional"]
GENRES = ["hip-hop", "R&B", "pop", "trap", "soul", "electronic", "afrobeats"]
GOALS = ["streams", "followers", "engagement", "sales", "awareness"]
AD_HOOKS = [
    "POV: your music goes viral overnight",
    "Stop scrolling — you need to hear this",
    "Finally. A beat that hits different.",
    "The sound everyone's been waiting for",
    "This song just broke my algorithm",
    "I wasn't ready for this drop",
    "Your playlist needs this right now",
    "Listen to this before it gets big",
]
CTA_LIST = ["Stream Now", "Listen Free", "Follow for More", "Drop in Comments", "Share if You Vibe"]
SOCIAL_HOOKS = [
    "Day {n} of dropping fire until I blow up",
    "No one talks about this part of being an indie artist",
    "How I went from 0 to 10k streams in 30 days",
    "The beat I made at 3am just hit different",
    "Behind the scenes: recording '{title}'",
    "Real talk — the music industry doesn't want you to know this",
    "POV: you discover your new favorite artist",
    "My most personal song yet just dropped",
]
TITLES = [
    "Midnight Sessions", "Neon Dreams", "Gold Rush", "Frequency", "Ascension",
    "Raw Footage", "Street Gospel", "Wavelength", "Overtime", "The Come Up",
]


def _random_social_post() -> dict:
    platform = random.choice(PLATFORMS)
    hook = random.choice(SOCIAL_HOOKS).format(
        n=random.randint(1, 100),
        title=random.choice(TITLES),
    )
    body = (
        f"Been working on this for {random.randint(2, 18)} months. "
        f"Every late night, every early morning went into this. "
        f"The {random.choice(GENRES)} vibes are real on this one. "
        f"Link in bio to stream."
    )
    cta = random.choice(CTA_LIST)
    hashtags = " ".join([
        f"#{random.choice(['NewMusic', 'IndieArtist', 'MusicProducer', 'HipHop', 'RnB', 'NewArtist'])}"
        for _ in range(random.randint(3, 6))
    ])
    return {
        "platform": platform,
        "hook": hook,
        "body": body,
        "cta": cta,
        "hashtags": hashtags,
        "text": f"{hook}\n\n{body}\n\n{cta}\n\n{hashtags}",
    }


def _random_ad_creative() -> dict:
    platform = random.choice(["TikTok", "Meta", "YouTube"])
    hook = random.choice(AD_HOOKS)
    cta = random.choice(CTA_LIST)
    product = f"{random.choice(TITLES)} {random.choice(['EP', 'Single', 'Album', 'Tape'])}"
    goal = random.choice(GOALS)
    audience = random.sample(
        ["hip-hop fans", "music lovers", "playlist listeners", "R&B fans",
         "urban culture", "trap enthusiasts", "indie music fans"], k=2
    )
    return {
        "platform": platform,
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
    bpm = random.randint(70, 160)
    key = random.choice(["C minor", "G major", "F# minor", "Bb major", "D minor", "A major"])
    lyric_hook = random.choice([
        f"I been on the grind since day one, nobody saw me coming",
        f"We made it out, we ain't going back to nothing",
        f"Every night I pray that I can find my way",
        f"The city lights reflect in my eyes at midnight",
        f"They never believed until the whole world was watching",
    ])
    return {
        "title": title,
        "genre": genre,
        "tone": tone,
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
    strategy = random.choice([
        "pitch to editorial playlists 7 days before release",
        "pre-save campaign starting 21 days out",
        "release day social blitz across all platforms",
        "TikTok pre-release audio leak strategy",
    ])
    return {
        "title": title,
        "release_days_out": release_in,
        "platforms": platforms,
        "strategy": strategy,
        "text": (
            f"Release '{title}' in {release_in} days on {', '.join(platforms)}. "
            f"Strategy: {strategy}."
        ),
    }


def generate_synthetic_samples(path: str, n: int = 500):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    generators = [
        _random_social_post,
        _random_ad_creative,
        _random_daw_content,
        _random_distribution_plan,
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
