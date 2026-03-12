from __future__ import annotations
import math
from typing import Optional


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.replace("0x", "").replace("#", "")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def animated_gradient_bg(w: int, h: int, dur: float, color1: str, color2: str, fps: int = 30) -> str:
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    return (
        f"geq=r='clip({r1}+(({r2}-{r1})*((Y+T*80)/{h}))\\,0\\,255)'"
        f":g='clip({g1}+(({g2}-{g1})*((Y+T*80)/{h}))\\,0\\,255)'"
        f":b='clip({b1}+(({b2}-{b1})*((Y+T*80)/{h}))\\,0\\,255)'"
    )


def radial_gradient_bg(w: int, h: int, color1: str, color2: str) -> str:
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    cx, cy = w // 2, h // 2
    max_dist = math.sqrt(cx * cx + cy * cy)
    return (
        f"geq=r='clip({r1}+({r2}-{r1})*hypot(X-{cx}\\,Y-{cy})/{max_dist:.0f}\\,0\\,255)'"
        f":g='clip({g1}+({g2}-{g1})*hypot(X-{cx}\\,Y-{cy})/{max_dist:.0f}\\,0\\,255)'"
        f":b='clip({b1}+({b2}-{b1})*hypot(X-{cx}\\,Y-{cy})/{max_dist:.0f}\\,0\\,255)'"
    )


def wave_gradient_bg(w: int, h: int, color1: str, color2: str) -> str:
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    return (
        f"geq=r='clip({r1}+({r2}-{r1})*(0.5+0.5*sin(Y/{h}*6.28+T*2+X/{w}*3.14))\\,0\\,255)'"
        f":g='clip({g1}+({g2}-{g1})*(0.5+0.5*sin(Y/{h}*6.28+T*2.5+X/{w}*3.14))\\,0\\,255)'"
        f":b='clip({b1}+({b2}-{b1})*(0.5+0.5*sin(Y/{h}*6.28+T*3+X/{w}*3.14))\\,0\\,255)'"
    )


def plasma_bg(w: int, h: int, color1: str, color2: str) -> str:
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    return (
        f"geq=r='clip({r1}+({r2}-{r1})*(0.5+0.25*sin(X/40+T*2)+0.25*cos(Y/30+T*1.5))\\,0\\,255)'"
        f":g='clip({g1}+({g2}-{g1})*(0.5+0.25*cos(X/35+T*1.7)+0.25*sin(Y/45+T*2.2))\\,0\\,255)'"
        f":b='clip({b1}+({b2}-{b1})*(0.5+0.25*sin((X+Y)/50+T*1.3)+0.25*cos(X/25-T*1.8))\\,0\\,255)'"
    )


def aurora_bg(w: int, h: int) -> str:
    return (
        f"geq=r='clip(10+20*sin(X/100+T*0.5)+15*cos(Y/80+T*0.3)\\,0\\,255)'"
        f":g='clip(30+60*sin(Y/{h}*3.14+T*0.7)*cos(X/200+T*0.4)+40*sin(X/150+T*0.6)\\,0\\,255)'"
        f":b='clip(60+80*sin(Y/{h}*3.14+T*0.5)*sin(X/120+T*0.8)+50*cos(Y/100+T*0.9)\\,0\\,255)'"
    )


def vignette_filter(intensity: float = 0.4) -> str:
    angle = max(0.1, min(1.0, intensity))
    return f"vignette=angle={angle}*PI/4"


def film_grain(amount: int = 15) -> str:
    amt = max(1, min(50, amount))
    return f"noise=alls={amt}:allf=t"


def color_grade_cinematic() -> str:
    return "curves=r='0/0 0.2/0.15 0.5/0.45 0.8/0.78 1/0.95':g='0/0 0.2/0.18 0.5/0.48 0.8/0.82 1/1':b='0/0.05 0.2/0.22 0.5/0.55 0.8/0.8 1/0.9'"


def color_grade_warm() -> str:
    return "colorbalance=rs=0.1:gs=0.05:bs=-0.1:rm=0.08:gm=0.03:bm=-0.05"


def color_grade_cool() -> str:
    return "colorbalance=rs=-0.1:gs=-0.02:bs=0.15:rm=-0.05:gm=0.02:bm=0.1"


def color_grade_neon() -> str:
    return "eq=saturation=1.6:contrast=1.2:brightness=0.02"


def color_grade_vintage() -> str:
    return "curves=r='0/0.05 0.3/0.28 0.7/0.72 1/0.9':g='0/0.02 0.3/0.25 0.7/0.65 1/0.85':b='0/0.1 0.3/0.22 0.7/0.55 1/0.75'"


def glow_effect(radius: int = 20, strength: float = 0.3) -> str:
    return f"split[main][glow];[glow]gblur=sigma={radius},eq=brightness=0.1[glowed];[main][glowed]blend=all_mode=screen:all_opacity={strength}"


def zoom_pulse(speed: float = 0.02) -> str:
    return f"zoompan=z='1+{speed}*sin(2*PI*on/90)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={{w}}x{{h}}:fps=30"


def slow_zoom_in(start: float = 1.0, end: float = 1.15, dur: float = 8.0) -> str:
    frames = int(dur * 30)
    step = (end - start) / frames
    return f"zoompan=z='min({end}\\,{start}+on*{step:.6f})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={{w}}x{{h}}:fps=30"


def slow_pan_down(amount: float = 0.1) -> str:
    return f"zoompan=z=1.1:x='iw/2-(iw/zoom/2)':y='min(ih*{amount}\\,on*2)':d=1:s={{w}}x{{h}}:fps=30"


def text_glow(fontfile: str, text: str, color: str, size: int, x: str, y: str, glow_color: str = "0xffffff", enable: str = "") -> str:
    enable_str = f":enable='{enable}'" if enable else ""
    parts = []
    for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, -3), (0, 3), (-3, 0), (3, 0)]:
        gx = f"({x})+{offset[0]}" if isinstance(x, str) and not x.isdigit() else f"{x}+{offset[0]}" if isinstance(x, str) else str(int(x) + offset[0])
        gy = f"({y})+{offset[1]}" if isinstance(y, str) and not y.isdigit() else f"{y}+{offset[1]}" if isinstance(y, str) else str(int(y) + offset[1])
        parts.append(
            f"drawtext=fontfile={fontfile}:text='{text}':fontcolor={glow_color}@0.3:fontsize={size}:x={gx}:y={gy}{enable_str}"
        )
    parts.append(
        f"drawtext=fontfile={fontfile}:text='{text}':fontcolor={color}:fontsize={size}:x={x}:y={y}{enable_str}"
    )
    return ",".join(parts)


def animated_border(w: int, h: int, color: str, thickness: int = 4) -> str:
    return (
        f"drawbox=x=0:y=0:w={w}:h={thickness}:color={color}@0.8:t=fill,"
        f"drawbox=x=0:y={h - thickness}:w={w}:h={thickness}:color={color}@0.8:t=fill,"
        f"drawbox=x=0:y=0:w={thickness}:h={h}:color={color}@0.8:t=fill,"
        f"drawbox=x={w - thickness}:y=0:w={thickness}:h={h}:color={color}@0.8:t=fill"
    )


def corner_accents(w: int, h: int, color: str, size: int = 60, thickness: int = 3) -> str:
    parts = []
    parts.append(f"drawbox=x=20:y=20:w={size}:h={thickness}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x=20:y=20:w={thickness}:h={size}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x={w - 20 - size}:y=20:w={size}:h={thickness}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x={w - 20 - thickness}:y=20:w={thickness}:h={size}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x=20:y={h - 20 - thickness}:w={size}:h={thickness}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x=20:y={h - 20 - size}:w={thickness}:h={size}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x={w - 20 - size}:y={h - 20 - thickness}:w={size}:h={thickness}:color={color}@0.7:t=fill")
    parts.append(f"drawbox=x={w - 20 - thickness}:y={h - 20 - size}:w={thickness}:h={size}:color={color}@0.7:t=fill")
    return ",".join(parts)


def scan_line_overlay(h: int, spacing: int = 4, opacity: float = 0.08) -> str:
    return f"geq=lum='lum(X\\,Y)*(1-{opacity}*(1-mod(Y\\,{spacing})/({spacing}-1)))':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"


def breathing_brightness(speed: float = 1.0, amount: float = 0.03) -> str:
    return f"eq=brightness='{amount}*sin(2*PI*{speed}*t)'"


def letterbox(w: int, h: int, bar_ratio: float = 0.12) -> str:
    bar_h = int(h * bar_ratio)
    return (
        f"drawbox=x=0:y=0:w={w}:h={bar_h}:color=black@0.85:t=fill,"
        f"drawbox=x=0:y={h - bar_h}:w={w}:h={bar_h}:color=black@0.85:t=fill"
    )


def progress_bar(w: int, h: int, color: str, height: int = 4) -> str:
    y = h - height - 2
    return (
        f"drawbox=x=0:y={y}:w={w}:h={height}:color=0x333333@0.5:t=fill,"
        f"drawbox=x=0:y={y}:w='t/duration*{w}':h={height}:color={color}@0.8:t=fill"
    )


# ============================================================================
# ADVANCED CINEMATIC EFFECTS
# ============================================================================

def chromatic_aberration(offset: int = 3) -> str:
    """RGB channel split effect for psychedelic/glitch aesthetic."""
    off = max(1, min(15, offset))
    return (
        f"[in]split=3[r][g][b];"
        f"[r]lutrgb=g=0:b=0,pad=iw+{2*off}:ih:{off}:0[rp];"
        f"[g]lutrgb=r=0:b=0[gp];"
        f"[b]lutrgb=r=0:g=0,pad=iw+{2*off}:ih:0:0[bp];"
        f"[rp][gp]blend=all_mode=addition:all_opacity=1[rg];"
        f"[rg][bp]blend=all_mode=addition:all_opacity=1[ca];"
        f"[ca]crop=iw-{2*off}:ih:{off}:0"
    )


def vhs_glitch(intensity: float = 0.4, fps: int = 30) -> str:
    """VHS tape glitch — scan jitter and horizontal distortion."""
    intens = max(0.1, min(1.0, intensity))
    amp = int(intens * 20)
    return (
        f"geq=lum='if(gt(random(0)\\,{1-intens*0.1})\\,lum(X\\,Y)\\,lum(X+{amp}*(random(1)-0.5)\\,Y))'"
        f":cb='cb(X\\,Y)':cr='cr(X\\,Y)',"
        f"noise=alls={int(intens*8)}:allf=t,"
        f"hue=s={1-intens*0.3}"
    )


def bloom_glow(radius: int = 25, intensity: float = 0.45) -> str:
    """Bloom/glow effect — highlights bleed into dark areas."""
    r = max(5, min(60, radius))
    i = max(0.1, min(0.9, intensity))
    return (
        f"split[orig][blur];"
        f"[blur]gblur=sigma={r},curves=r='0/0 0.5/0.7 1/1':g='0/0 0.5/0.7 1/1':b='0/0 0.5/0.7 1/1'[bloomed];"
        f"[orig][bloomed]blend=all_mode=screen:all_opacity={i}"
    )


def zoom_blur(amount: float = 0.06) -> str:
    """Radial zoom blur — cinematic speed effect."""
    a = max(0.01, min(0.2, amount))
    return f"zoompan=z='1+{a}*sin(on/30*PI)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=iw:ih:fps=30"


def holographic_shimmer(speed: float = 2.0, hue_range: float = 0.3) -> str:
    """Holographic rainbow shimmer — cycling hue shift over time."""
    s = max(0.5, min(6.0, speed))
    h = max(0.1, min(1.0, hue_range))
    return (
        f"hue=h='{h*360}*sin(2*PI*{s}*t)':s='1+0.3*cos(2*PI*{s}*t)',"
        f"eq=saturation=1.4:contrast=1.05"
    )


def double_exposure_blend(opacity: float = 0.5) -> str:
    """Double exposure blend — overlaid image transparency for artistic effect."""
    o = max(0.1, min(0.9, opacity))
    return f"blend=all_mode=overlay:all_opacity={o}"


def scanlines(h: int, spacing: int = 3, opacity: float = 0.12, animated: bool = False) -> str:
    """CRT scan line overlay. Animated version scrolls lines downward."""
    if animated:
        return f"geq=lum='lum(X\\,Y)*(1-{opacity}*lt(mod(Y+t*60\\,{spacing})\\,1))':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"
    return f"geq=lum='lum(X\\,Y)*(1-{opacity}*(1-mod(Y\\,{spacing})/({spacing}-1)))':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"


def pixel_sort(threshold: float = 0.5, axis: str = 'h') -> str:
    """Pixel sorting glitch effect — sorts pixels by brightness in a direction."""
    t = max(0.1, min(0.9, threshold))
    if axis == 'v':
        return f"geq=lum='if(gt(lum(X\\,Y)\\,{t*255})\\,lum(X\\,Y)\\,lum(X\\,max(0\\,Y-1)))':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"
    return f"geq=lum='if(gt(lum(X\\,Y)\\,{t*255})\\,lum(X\\,Y)\\,lum(max(0\\,X-1)\\,Y))':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"


def color_grade_neon_underground() -> str:
    """Deep purple-to-neon-pink gradient grade. Underground club aesthetic."""
    return (
        "curves=r='0/0 0.2/0.3 0.6/0.8 1/1':g='0/0 0.3/0.1 0.7/0.6 1/0.9':b='0/0.1 0.3/0.4 0.7/0.9 1/1',"
        "eq=saturation=1.7:contrast=1.15:brightness=-0.02"
    )


def color_grade_golden_hour() -> str:
    """Warm cinematic golden-hour look. Sunset release aesthetic."""
    return (
        "curves=r='0/0.05 0.3/0.4 0.7/0.85 1/1':g='0/0 0.3/0.28 0.7/0.72 1/0.92':b='0/0 0.3/0.18 0.7/0.55 1/0.75',"
        "colorbalance=rs=0.15:gs=0.05:bs=-0.15:rm=0.1:gm=0.02:bm=-0.08,"
        "eq=saturation=1.2"
    )


def color_grade_dark_academia() -> str:
    """Desaturated warm tones with lifted shadows. Moody intellectual aesthetic."""
    return (
        "curves=r='0/0.08 0.3/0.32 0.7/0.75 1/0.95':g='0/0.05 0.3/0.28 0.7/0.7 1/0.9':b='0/0.02 0.3/0.2 0.7/0.6 1/0.82',"
        "eq=saturation=0.75:contrast=1.1"
    )


def color_grade_hyperpop() -> str:
    """Oversaturated, blown highlights, neon pastels. Hyperpop chaos."""
    return (
        "eq=saturation=2.2:contrast=1.3:brightness=0.05,"
        "curves=r='0/0 0.4/0.55 0.8/0.95 1/1':g='0/0 0.4/0.45 0.8/0.85 1/1':b='0/0 0.4/0.6 0.8/1.0 1/1'"
    )


def color_grade_afro_vibes() -> str:
    """Warm, saturated, rich tones. Afrobeats / African sun energy."""
    return (
        "colorbalance=rs=0.1:gs=0.08:bs=-0.12:rm=0.08:gm=0.05:bm=-0.1,"
        "eq=saturation=1.4:contrast=1.08:brightness=0.02,"
        "curves=r='0/0 0.3/0.35 0.7/0.8 1/1'"
    )


def color_grade_lofi_chill() -> str:
    """Faded, warm, low-contrast tape look. Lo-fi chill aesthetic."""
    return (
        "curves=r='0/0.08 0.3/0.35 0.7/0.72 1/0.9':g='0/0.06 0.3/0.3 0.7/0.65 1/0.85':b='0/0.12 0.3/0.3 0.7/0.6 1/0.78',"
        "eq=saturation=0.65:contrast=0.92,"
        "noise=alls=5:allf=t"
    )


def color_grade_drill_streets() -> str:
    """Cold, desaturated, high-contrast. UK/NY drill street aesthetic."""
    return (
        "colorbalance=rs=-0.08:gs=-0.05:bs=0.12:rm=-0.05:gm=-0.02:bm=0.08,"
        "eq=saturation=0.55:contrast=1.35:brightness=-0.03"
    )


def color_grade_techno_industrial() -> str:
    """Stark monochrome teal-shift. Industrial techno aesthetic."""
    return (
        "hue=s=0.3,"
        "colorbalance=rs=-0.1:gs=0:bs=0.2,"
        "eq=contrast=1.4:brightness=-0.05"
    )


def text_animation_typewriter(fontfile: str, text: str, color: str, size: int,
                               x: str, y: str, start: float = 0.3, speed: int = 8) -> str:
    """Typewriter text reveal — characters appear one at a time."""
    chars = min(len(text), 80)
    parts = []
    for i in range(1, chars + 1):
        t_start = start + i * (1 / speed)
        t_end = start + chars * (1 / speed) + 3.0
        sub = text[:i].replace("'", "\\'")
        parts.append(
            f"drawtext=fontfile={fontfile}:text='{sub}':fontcolor={color}:fontsize={size}"
            f":x={x}:y={y}:enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'"
        )
    return ",".join(parts)


def text_animation_glitch_reveal(fontfile: str, text: str, color: str, size: int,
                                  x: str, y: str, start: float = 0.3) -> str:
    """Glitch-style text reveal — text flickers in with offset layers."""
    safe_text = text.replace("'", "\\'")
    parts = []
    for i, (ox, oy, op, t_off) in enumerate([(-3, 0, 0.5, 0), (3, 0, 0.5, 0.05), (0, 0, 1.0, 0.1)]):
        xstr = f"({x})+{ox}" if ox != 0 else x
        ystr = f"({y})+{oy}" if oy != 0 else y
        parts.append(
            f"drawtext=fontfile={fontfile}:text='{safe_text}':fontcolor={color}@{op}"
            f":fontsize={size}:x={xstr}:y={ystr}:enable='gte(t\\,{start+t_off:.2f})'"
        )
    return ",".join(parts)


XFADE_TRANSITIONS = [
    "fade", "fadeblack", "fadewhite", "wipeleft", "wiperight",
    "wipeup", "wipedown", "slideleft", "slideright", "slideup",
    "slidedown", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circlecrop", "circleopen", "circleclose", "dissolve",
    "pixelize", "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hblur", "radial", "zoomin",
]
