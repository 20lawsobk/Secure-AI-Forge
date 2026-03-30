import { useState, useRef, useEffect, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import {
  Film, Sparkles, Play, Download, RefreshCw, AlertCircle,
  Wand2, ChevronRight, Clock, Zap, Layers, Radio,
  Cpu, BarChart3, Check, Edit3
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const BASE = "/api";

// ── Types ────────────────────────────────────────────────────────────────────

interface Scene {
  type: "hook" | "build" | "body" | "drop" | "cta" | "outro";
  text: string;
}

interface GenerateResponse {
  job_id: string;
  status: string;
  genre_detected: string;
  tone_used: string;
  source: string;
  duration: number;
  aspect_ratio: string;
  scenes: Scene[];
  poll_url: string;
  template: string;
}

interface PollResponse {
  status: "pending" | "running" | "done" | "error";
  url?: string;
  filename?: string;
  width?: number;
  height?: number;
  duration?: number;
  scenes_rendered?: number;
  render_ms?: number;
  aspect_ratio?: string;
  genre_detected?: string;
  tone_used?: string;
  source?: string;
  scenes?: Scene[];
  error?: string;
}

// ── Constants ────────────────────────────────────────────────────────────────

const PLATFORMS = [
  { value: "tiktok", label: "TikTok", ratio: "9:16", icon: "▶" },
  { value: "instagram_reels", label: "Instagram Reels", ratio: "9:16", icon: "◉" },
  { value: "instagram", label: "Instagram Feed", ratio: "1:1", icon: "◉" },
  { value: "youtube_shorts", label: "YouTube Shorts", ratio: "9:16", icon: "▷" },
  { value: "youtube", label: "YouTube", ratio: "16:9", icon: "▷" },
  { value: "twitter", label: "X (Twitter)", ratio: "16:9", icon: "✕" },
  { value: "linkedin", label: "LinkedIn", ratio: "16:9", icon: "in" },
  { value: "facebook_reels", label: "Facebook Reels", ratio: "9:16", icon: "f" },
];

const GENRES = [
  "trap", "drill", "hiphop", "rnb", "soul", "jazz",
  "pop", "indie", "acoustic", "lofi",
  "afrobeats", "reggaeton", "latin",
  "electronic", "hyperpop",
];

const TONES = [
  { value: "energetic", label: "Energetic" },
  { value: "hype", label: "Hype" },
  { value: "edgy", label: "Edgy" },
  { value: "chill", label: "Chill" },
  { value: "emotional", label: "Emotional" },
  { value: "professional", label: "Professional" },
  { value: "playful", label: "Playful" },
  { value: "inspirational", label: "Inspirational" },
  { value: "promotional", label: "Promotional" },
  { value: "serious", label: "Serious" },
];

const GOALS = [
  { value: "growth", label: "Growth & Viral" },
  { value: "conversion", label: "Conversion" },
  { value: "engagement", label: "Engagement" },
  { value: "awareness", label: "Awareness" },
  { value: "streams", label: "Streams" },
  { value: "sales", label: "Sales" },
];

const SCENE_LABELS: Record<string, { color: string; label: string }> = {
  hook:  { color: "text-primary bg-primary/10 border-primary/30", label: "HOOK" },
  build: { color: "text-blue-400 bg-blue-400/10 border-blue-400/30", label: "BUILD" },
  body:  { color: "text-gray-300 bg-white/5 border-white/15", label: "BODY" },
  drop:  { color: "text-fuchsia-400 bg-fuchsia-400/10 border-fuchsia-400/30", label: "DROP" },
  cta:   { color: "text-amber-400 bg-amber-400/10 border-amber-400/30", label: "CTA" },
  outro: { color: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30", label: "OUTRO" },
};

// ── Render progress estimation ────────────────────────────────────────────────

function useRenderProgress(isRendering: boolean, sceneCount: number) {
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!isRendering) {
      setProgress(0);
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    setProgress(2);
    const estimatedMs = sceneCount * 22_000;
    const tick = 800;
    let elapsed = 0;
    intervalRef.current = setInterval(() => {
      elapsed += tick;
      const pct = Math.min(92, Math.round((elapsed / estimatedMs) * 100));
      setProgress(pct);
    }, tick);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRendering, sceneCount]);

  return progress;
}

// ── DNA Indicator ─────────────────────────────────────────────────────────────

function DnaBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground w-20 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(value * 100)}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
      <span className="text-xs text-muted-foreground w-8 text-right">{Math.round(value * 100)}</span>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function VideoStudio() {
  const { toast } = useToast();
  const { adminKey } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);

  // Form state
  const [idea, setIdea] = useState("");
  const [platform, setPlatform] = useState("tiktok");
  const [genre, setGenre] = useState("trap");
  const [tone, setTone] = useState("energetic");
  const [goal, setGoal] = useState("growth");
  const [artistName, setArtistName] = useState("");
  const [duration, setDuration] = useState<number | "">("");

  // Pipeline state
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [renderResult, setRenderResult] = useState<PollResponse | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [generationMeta, setGenerationMeta] = useState<Partial<GenerateResponse> | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progress = useRenderProgress(isPolling, scenes.length);

  // ── API calls ────────────────────────────────────────────────────────────

  const authHeaders = useCallback((): Record<string, string> => {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (adminKey) h["X-Admin-Key"] = adminKey;
    return h;
  }, [adminKey]);

  const generateMut = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${BASE}/video/generate-ai`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          idea: idea.trim(),
          platform,
          genre,
          tone,
          goal,
          artist_name: artistName.trim(),
          duration: duration || 0,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Generation failed: ${res.status}`);
      }
      return res.json() as Promise<GenerateResponse>;
    },
    onSuccess: (data) => {
      setScenes(data.scenes || []);
      setGenerationMeta(data);
      setJobId(null);
      setRenderResult(null);
      toast({ title: "Scenes generated", description: `${data.scenes.length} scenes ready — edit if needed, then render` });
    },
    onError: (e: Error) => {
      toast({ title: "Generation failed", description: e.message, variant: "destructive" });
    },
  });

  const renderMut = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${BASE}/video/generate-ai`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          idea: idea.trim(),
          platform,
          genre,
          tone,
          goal,
          artist_name: artistName.trim(),
          duration: duration || 0,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Render failed: ${res.status}`);
      }
      return res.json() as Promise<GenerateResponse>;
    },
    onSuccess: (data) => {
      setJobId(data.job_id);
      setIsPolling(true);
      setRenderResult(null);
      toast({ title: "Rendering started", description: "AI is generating your video…" });
    },
    onError: (e: Error) => {
      toast({ title: "Render failed", description: e.message, variant: "destructive" });
    },
  });

  // Poll job status
  useEffect(() => {
    if (!jobId || !isPolling) return;
    if (pollRef.current) clearInterval(pollRef.current);

    const poll = async () => {
      try {
        const res = await fetch(`${BASE}/video-job/${jobId}`, { headers: authHeaders() });
        if (!res.ok) return;
        const data: PollResponse = await res.json();
        if (data.status === "done" || data.status === "error") {
          clearInterval(pollRef.current!);
          setIsPolling(false);
          setRenderResult(data);
          if (data.status === "done") {
            toast({ title: "Video ready!", description: "Your video has been rendered." });
          } else {
            toast({ title: "Render failed", description: data.error, variant: "destructive" });
          }
        }
      } catch {
        // network blip — keep polling
      }
    };

    pollRef.current = setInterval(poll, 3000);
    poll();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId, isPolling, authHeaders, toast]);

  // ── Scene editing ─────────────────────────────────────────────────────────

  const startEdit = (idx: number) => {
    setEditingIdx(idx);
    setEditText(scenes[idx].text);
  };

  const commitEdit = () => {
    if (editingIdx === null) return;
    setScenes(prev => prev.map((s, i) => i === editingIdx ? { ...s, text: editText } : s));
    setEditingIdx(null);
  };

  // ── Derived ──────────────────────────────────────────────────────────────

  const currentPlatform = PLATFORMS.find(p => p.value === platform) || PLATFORMS[0];
  const isPortrait = currentPlatform.ratio === "9:16";
  const hasScenes = scenes.length > 0;
  const isGenerating = generateMut.isPending;
  const isRendering = renderMut.isPending || isPolling;
  const videoDone = renderResult?.status === "done";

  const videoUrl = renderResult?.url
    ? `${BASE.replace("/api", "")}${renderResult.url}`
    : null;

  // Rough DNA visualisation from genre/tone (client-side approximation)
  const dnaApprox = {
    energy: { trap: 0.90, drill: 0.85, hiphop: 0.80, rnb: 0.50, soul: 0.40, jazz: 0.30, pop: 0.70, afrobeats: 0.85, reggaeton: 0.85, latin: 0.80, lofi: 0.20, indie: 0.38, acoustic: 0.28, electronic: 0.92, hyperpop: 1.00 } as Record<string, number>,
    darkness: { trap: 0.88, drill: 0.92, hiphop: 0.75, rnb: 0.60, soul: 0.60, jazz: 0.70, pop: 0.30, afrobeats: 0.45, reggaeton: 0.40, latin: 0.40, lofi: 0.55, indie: 0.42, acoustic: 0.32, electronic: 0.82, hyperpop: 0.25 } as Record<string, number>,
    saturation: { trap: 0.80, drill: 0.45, hiphop: 0.70, rnb: 0.60, soul: 0.55, jazz: 0.40, pop: 0.90, afrobeats: 0.95, reggaeton: 0.90, latin: 0.88, lofi: 0.38, indie: 0.48, acoustic: 0.40, electronic: 0.88, hyperpop: 1.00 } as Record<string, number>,
  };

  const energy = Math.min(1, (dnaApprox.energy[genre] ?? 0.65) + (tone === "hype" || tone === "energetic" ? 0.1 : tone === "chill" ? -0.2 : 0));
  const darkness = Math.min(1, (dnaApprox.darkness[genre] ?? 0.60) + (tone === "edgy" ? 0.1 : tone === "playful" ? -0.15 : 0));
  const saturation = Math.min(1, (dnaApprox.saturation[genre] ?? 0.70) + (tone === "playful" ? 0.1 : tone === "professional" ? -0.15 : 0));

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-fuchsia-600 flex items-center justify-center shadow-lg shadow-primary/20">
            <Film className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-display font-bold text-white">Video Studio</h1>
            <p className="text-sm text-muted-foreground">Generative AI rendering — every video uniquely derived from your content</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {generationMeta && (
            <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5 text-xs font-mono">
              <Cpu className="w-3 h-3 mr-1" /> {generationMeta.source === "ai_model" ? "model" : "script"} • {generationMeta.aspect_ratio}
            </Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">

        {/* ── Left: Configuration ─────────────────────────────────────────── */}
        <div className="xl:col-span-4 space-y-4">
          <Card className="glass-panel p-5 border-white/10 space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Wand2 className="w-4 h-4 text-primary" /> Concept
            </h3>

            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Idea / Topic</label>
              <Textarea
                placeholder="e.g. debut single dropping Friday, new drop announcement, behind the scenes…"
                className="bg-black/50 border-white/10 text-white min-h-[80px] resize-none text-sm focus:border-primary/50"
                value={idea}
                onChange={e => setIdea(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Platform</label>
                <Select value={platform} onValueChange={setPlatform}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white text-sm h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {PLATFORMS.map(p => (
                      <SelectItem key={p.value} value={p.value}>
                        <span className="flex items-center gap-2">
                          {p.label}
                          <span className="text-xs text-muted-foreground">{p.ratio}</span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Genre</label>
                <Select value={genre} onValueChange={setGenre}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white text-sm h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {GENRES.map(g => (
                      <SelectItem key={g} value={g} className="capitalize">{g}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Tone</label>
                <Select value={tone} onValueChange={setTone}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white text-sm h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {TONES.map(t => (
                      <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Goal</label>
                <Select value={goal} onValueChange={setGoal}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white text-sm h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {GOALS.map(g => (
                      <SelectItem key={g.value} value={g.value}>{g.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Artist Name</label>
                <Input
                  placeholder="Your name…"
                  className="bg-black/50 border-white/10 text-white text-sm h-9 focus:border-primary/50"
                  value={artistName}
                  onChange={e => setArtistName(e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Duration (s)</label>
                <Input
                  type="number"
                  placeholder="Auto"
                  min={6}
                  max={120}
                  className="bg-black/50 border-white/10 text-white text-sm h-9 focus:border-primary/50"
                  value={duration}
                  onChange={e => setDuration(e.target.value ? Number(e.target.value) : "")}
                />
              </div>
            </div>

            <Button
              className="w-full bg-gradient-to-r from-primary to-fuchsia-600 hover:opacity-90 text-white h-10 font-semibold shadow-lg shadow-primary/20"
              onClick={() => generateMut.mutate()}
              disabled={!idea.trim() || isGenerating || isRendering}
            >
              {isGenerating ? (
                <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" /> Generating scenes…</>
              ) : (
                <><Sparkles className="w-4 h-4 mr-2" /> Generate Scenes</>
              )}
            </Button>
          </Card>

          {/* Visual DNA Card */}
          <Card className="glass-panel p-5 border-white/10 space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-fuchsia-400" /> Visual DNA
              <span className="text-xs text-muted-foreground font-normal ml-auto">derived from genre + tone</span>
            </h3>
            <DnaBar label="Energy" value={energy} color="bg-primary" />
            <DnaBar label="Darkness" value={darkness} color="bg-slate-400" />
            <DnaBar label="Saturation" value={saturation} color="bg-fuchsia-500" />
            <p className="text-xs text-muted-foreground pt-1 leading-relaxed">
              These values drive background type, colour palette, effects, and typography — regenerated fresh for every render. No templates.
            </p>
          </Card>
        </div>

        {/* ── Center: Scene Editor ────────────────────────────────────────── */}
        <div className="xl:col-span-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-400" /> Scenes
              {hasScenes && <span className="text-xs text-muted-foreground font-normal">({scenes.length} scenes — click to edit)</span>}
            </h3>
            {hasScenes && (
              <Badge variant="outline" className="text-xs border-white/10 text-muted-foreground">
                <Clock className="w-3 h-3 mr-1" />
                {generationMeta?.duration ? `${generationMeta.duration}s` : "auto"}
              </Badge>
            )}
          </div>

          <AnimatePresence mode="wait">
            {!hasScenes && !isGenerating ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex-1 min-h-[380px] border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-muted-foreground gap-3"
              >
                <Film className="w-10 h-10 opacity-20" />
                <p className="text-sm">Fill in your concept and click Generate Scenes</p>
              </motion.div>
            ) : isGenerating ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex-1 min-h-[380px] glass-panel rounded-2xl flex flex-col items-center justify-center gap-4 text-primary"
              >
                <div className="w-10 h-10 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
                <p className="animate-pulse font-medium text-sm">Model generating scenes…</p>
              </motion.div>
            ) : (
              <motion.div
                key="scenes"
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-3"
              >
                {scenes.map((scene, idx) => {
                  const meta = SCENE_LABELS[scene.type] || SCENE_LABELS.body;
                  const isEditing = editingIdx === idx;
                  return (
                    <motion.div
                      key={idx}
                      layout
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                    >
                      <Card className="glass-panel border-white/10 p-4 group hover:border-white/20 transition-all">
                        <div className="flex items-start gap-3">
                          <div className="flex flex-col items-center gap-1 pt-0.5 shrink-0">
                            <span className={`text-[10px] font-bold border rounded px-1.5 py-0.5 ${meta.color}`}>
                              {meta.label}
                            </span>
                            <span className="text-xs text-muted-foreground">{idx + 1}</span>
                          </div>
                          <div className="flex-1 min-w-0">
                            {isEditing ? (
                              <div className="space-y-2">
                                <Textarea
                                  autoFocus
                                  value={editText}
                                  onChange={e => setEditText(e.target.value)}
                                  className="bg-black/50 border-white/20 text-white text-sm resize-none min-h-[60px] focus:border-primary/50"
                                  onKeyDown={e => { if (e.key === "Enter" && e.metaKey) commitEdit(); if (e.key === "Escape") setEditingIdx(null); }}
                                />
                                <div className="flex gap-2">
                                  <Button size="sm" className="h-7 text-xs bg-primary/20 hover:bg-primary/30 text-primary border border-primary/30" onClick={commitEdit}>
                                    <Check className="w-3 h-3 mr-1" /> Save
                                  </Button>
                                  <Button size="sm" variant="ghost" className="h-7 text-xs text-muted-foreground" onClick={() => setEditingIdx(null)}>
                                    Cancel
                                  </Button>
                                </div>
                              </div>
                            ) : (
                              <p
                                className="text-sm text-gray-200 leading-relaxed cursor-pointer hover:text-white transition-colors"
                                onClick={() => startEdit(idx)}
                              >
                                {scene.text}
                              </p>
                            )}
                          </div>
                          {!isEditing && (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-white shrink-0"
                              onClick={() => startEdit(idx)}
                            >
                              <Edit3 className="w-3.5 h-3.5" />
                            </Button>
                          )}
                        </div>
                      </Card>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Right: Render / Output ──────────────────────────────────────── */}
        <div className="xl:col-span-3 flex flex-col gap-4">
          {/* Render control */}
          <Card className="glass-panel p-5 border-white/10 space-y-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" /> Render
            </h3>

            {/* Platform preview badge */}
            <div className="flex items-center gap-2 p-3 bg-white/5 rounded-xl border border-white/10">
              <div className={`bg-white/10 rounded flex items-center justify-center text-white font-bold text-xs
                ${isPortrait ? "w-9 h-14" : "w-14 h-9"}`}>
                {currentPlatform.ratio}
              </div>
              <div>
                <p className="text-xs font-medium text-white">{currentPlatform.label}</p>
                <p className="text-xs text-muted-foreground">{currentPlatform.ratio} · {isPortrait ? "1080×1920" : currentPlatform.ratio === "1:1" ? "1080×1080" : "1920×1080"}</p>
              </div>
            </div>

            <Button
              className="w-full bg-gradient-to-r from-amber-500 to-orange-600 hover:opacity-90 text-white h-10 font-semibold shadow-lg shadow-amber-500/20 disabled:opacity-50"
              onClick={() => renderMut.mutate()}
              disabled={!idea.trim() || isRendering || isGenerating}
            >
              {isRendering ? (
                <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" /> Rendering…</>
              ) : (
                <><Radio className="w-4 h-4 mr-2" /> Render Video</>
              )}
            </Button>

            <p className="text-xs text-muted-foreground text-center">
              Visuals are generated fresh — colours, effects, and typography are computed directly from your genre, tone, and idea.
            </p>
          </Card>

          {/* Render progress */}
          <AnimatePresence>
            {isRendering && (
              <motion.div
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              >
                <Card className="glass-panel p-4 border-primary/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-primary">Rendering scenes via FFmpeg…</p>
                    <span className="text-xs text-muted-foreground">{progress}%</span>
                  </div>
                  <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-primary to-fuchsia-500 rounded-full"
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    AI scene builder deriving palette and effects from content DNA…
                  </p>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error state */}
          <AnimatePresence>
            {renderResult?.status === "error" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card className="glass-panel p-4 border-destructive/30 bg-destructive/5">
                  <div className="flex gap-2 items-start">
                    <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-destructive">Render failed</p>
                      <p className="text-xs text-muted-foreground mt-1">{renderResult.error}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Render metadata */}
          {videoDone && renderResult && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
              <Card className="glass-panel p-4 border-emerald-500/20 bg-emerald-500/5 space-y-2">
                <div className="flex items-center gap-2 text-emerald-400">
                  <Check className="w-4 h-4" />
                  <span className="text-sm font-medium">Render complete</span>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  <span>Resolution</span><span className="text-white">{renderResult.width}×{renderResult.height}</span>
                  <span>Duration</span><span className="text-white">{renderResult.duration}s</span>
                  <span>Scenes</span><span className="text-white">{renderResult.scenes_rendered}</span>
                  <span>Render time</span><span className="text-white">{renderResult.render_ms ? `${(renderResult.render_ms / 1000).toFixed(1)}s` : "—"}</span>
                </div>
              </Card>
            </motion.div>
          )}
        </div>
      </div>

      {/* ── Video Player ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {videoDone && videoUrl && (
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="mt-2"
          >
            <Card className="glass-panel border-white/10 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Play className="w-4 h-4 text-primary" /> Preview
                  <Badge variant="outline" className="text-xs text-emerald-400 border-emerald-400/30 bg-emerald-400/5 ml-2">
                    ai_derived
                  </Badge>
                </h3>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-8 text-xs border-white/10 text-white hover:bg-white/10 gap-1.5"
                    onClick={() => generateMut.mutate()}
                    disabled={isGenerating || isRendering}
                  >
                    <RefreshCw className="w-3 h-3" /> Regenerate
                  </Button>
                  <Button
                    size="sm"
                    className="h-8 text-xs bg-primary/20 hover:bg-primary/30 text-primary border border-primary/30 gap-1.5"
                    asChild
                  >
                    <a href={videoUrl} download={renderResult?.filename || "video.mp4"}>
                      <Download className="w-3 h-3" /> Download
                    </a>
                  </Button>
                </div>
              </div>

              <div className={`mx-auto overflow-hidden rounded-xl bg-black shadow-2xl shadow-black/60 ${isPortrait ? "max-w-[320px]" : "max-w-full"}`}>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  autoPlay
                  loop
                  className="w-full h-auto block"
                  style={{ aspectRatio: currentPlatform.ratio === "9:16" ? "9/16" : currentPlatform.ratio === "1:1" ? "1/1" : "16/9" }}
                />
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
