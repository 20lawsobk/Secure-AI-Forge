import { useState, useRef } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";
import {
  Link2,
  Search,
  ExternalLink,
  Sparkles,
  ArrowRight,
  Copy,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Music2,
  Tag,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const BASE = "/api";

interface InspectResult {
  raw_url: string;
  canonical_url: string;
  platform: string;
  platform_label: string;
  content_type: string;
  title: string;
  artist: string;
  album: string;
  label: string;
  description: string;
  genre: string;
  mood: string;
  bpm: number | null;
  key: string;
  release_year: number | null;
  intent: string;
  goal: string;
  content_themes: string[];
  topic_string: string;
  awareness_text: string;
  fetch_ok: boolean;
  error: string;
}

const PLATFORM_COLORS: Record<string, string> = {
  spotify: "bg-green-500/15 text-green-400 border-green-500/30",
  youtube: "bg-red-500/15 text-red-400 border-red-500/30",
  tiktok: "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
  instagram: "bg-pink-500/15 text-pink-400 border-pink-500/30",
  soundcloud: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  apple_music: "bg-pink-500/15 text-pink-400 border-pink-500/30",
  twitter: "bg-sky-500/15 text-sky-400 border-sky-500/30",
  linkedin: "bg-blue-600/15 text-blue-400 border-blue-500/30",
};

const EXAMPLE_URLS = [
  "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh",
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "https://www.tiktok.com/@khaby.lame",
  "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
];

function MetaRow({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number | null | undefined;
  accent?: string;
}) {
  if (!value && value !== 0) return null;
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-white/5 last:border-0">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider w-28 flex-shrink-0 pt-0.5">
        {label}
      </span>
      <span className={`text-sm text-white flex-1 ${accent ?? ""}`}>
        {String(value)}
      </span>
    </div>
  );
}

export default function UrlInspector() {
  const { toast } = useToast();
  const { adminKey } = useAuth();
  const [, navigate] = useLocation();

  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InspectResult | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const inspect = async (urlToInspect: string) => {
    const trimmed = urlToInspect.trim();
    if (!trimmed) {
      toast({ title: "Please enter a URL", variant: "destructive" });
      return;
    }

    setLoading(true);
    setResult(null);
    setFetchError(null);

    try {
      const headers: Record<string, string> = {};
      if (adminKey) headers["X-Admin-Key"] = adminKey;

      const res = await fetch(
        `${BASE}/url-parser/inspect?url=${encodeURIComponent(trimmed)}`,
        { headers },
      );

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        const msg =
          data?.detail || data?.error || `Request failed (${res.status})`;
        setFetchError(msg);
        return;
      }

      setResult(data as InspectResult);
    } catch (e) {
      setFetchError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    inspect(url);
  };

  const copy = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: "Copied to clipboard" });
  };

  const prefillGenerator = () => {
    if (!result?.topic_string) return;
    navigate(`/content?topic=${encodeURIComponent(result.topic_string)}`);
  };

  const platformBadgeClass =
    PLATFORM_COLORS[result?.platform ?? ""] ??
    "bg-white/10 text-muted-foreground border-white/10";

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto mb-10">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-fuchsia-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-primary/20">
          <Link2 className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl font-display font-bold text-white mb-3">
          URL Inspector
        </h1>
        <p className="text-muted-foreground text-lg">
          Paste any Spotify, TikTok, YouTube, or social link to preview exactly
          what signals the AI extracts before generating content.
        </p>
      </div>

      {/* Search bar */}
      <form onSubmit={handleSubmit}>
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
            <Input
              ref={inputRef}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://open.spotify.com/track/… or spotify:track:…"
              className="pl-10 bg-black/50 border-white/10 text-white h-12 text-base focus-visible:ring-primary"
            />
          </div>
          <Button
            type="submit"
            disabled={loading}
            className="h-12 px-6 bg-gradient-to-r from-primary to-fuchsia-600 hover:from-primary/90 hover:to-fuchsia-600/90 text-white shadow-lg shadow-primary/25 font-semibold"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                Inspect <ArrowRight className="w-4 h-4 ml-2" />
              </>
            )}
          </Button>
        </div>
      </form>

      {/* Example URLs */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-muted-foreground self-center mr-1">
          Try:
        </span>
        {EXAMPLE_URLS.map((ex) => (
          <button
            key={ex}
            onClick={() => {
              setUrl(ex);
              inspect(ex);
            }}
            className="text-xs text-primary/80 hover:text-primary bg-primary/5 hover:bg-primary/10 border border-primary/20 px-2.5 py-1 rounded-full transition-colors truncate max-w-[240px]"
          >
            {ex.replace("https://", "").slice(0, 40)}…
          </button>
        ))}
      </div>

      {/* Error state */}
      <AnimatePresence>
        {fetchError && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-start gap-3 bg-destructive/10 border border-destructive/30 rounded-xl p-4 text-destructive"
          >
            <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <p className="text-sm">{fetchError}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading skeleton */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center gap-4 min-h-[300px]"
          >
            <div className="w-10 h-10 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
            <p className="text-muted-foreground animate-pulse">
              Fetching & parsing URL…
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Status bar */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-3 flex-wrap">
                {result.platform_label && (
                  <Badge
                    className={`border text-xs font-semibold px-3 py-1 ${platformBadgeClass}`}
                  >
                    {result.platform_label}
                  </Badge>
                )}
                {result.content_type && (
                  <Badge
                    variant="outline"
                    className="border-white/10 text-muted-foreground capitalize text-xs"
                  >
                    {result.content_type}
                  </Badge>
                )}
                {result.fetch_ok ? (
                  <span className="flex items-center gap-1.5 text-xs text-green-400">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Fetched
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-xs text-amber-400">
                    <AlertCircle className="w-3.5 h-3.5" /> No HTML fetch
                    (signals from URL only)
                  </span>
                )}
                {result.error && (
                  <span className="text-xs text-destructive">{result.error}</span>
                )}
              </div>

              <Button
                onClick={prefillGenerator}
                size="sm"
                className="bg-gradient-to-r from-primary to-fuchsia-600 hover:from-primary/90 hover:to-fuchsia-600/90 text-white shadow-md shadow-primary/20 h-8 text-xs font-semibold"
              >
                <Sparkles className="w-3.5 h-3.5 mr-1.5" />
                Generate from this URL
              </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Content metadata */}
              <div className="space-y-6">
                <Card className="glass-panel p-6 border-white/10">
                  <div className="flex items-center gap-2 mb-4">
                    <Tag className="w-4 h-4 text-primary" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                      Content Metadata
                    </h3>
                  </div>
                  <div>
                    <MetaRow label="Topic" value={result.topic_string} accent="font-semibold text-primary" />
                    <MetaRow label="Title" value={result.title} />
                    <MetaRow label="Artist" value={result.artist} />
                    <MetaRow label="Album" value={result.album} />
                    <MetaRow label="Label" value={result.label} />
                    <MetaRow label="Description" value={result.description?.slice(0, 160) || ""} />
                  </div>
                </Card>

                <Card className="glass-panel p-6 border-white/10">
                  <div className="flex items-center gap-2 mb-4">
                    <Music2 className="w-4 h-4 text-fuchsia-400" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                      Music Signals
                    </h3>
                  </div>
                  <div>
                    <MetaRow label="Genre" value={result.genre} />
                    <MetaRow label="Mood" value={result.mood} />
                    <MetaRow label="BPM" value={result.bpm} />
                    <MetaRow label="Key" value={result.key} />
                    <MetaRow label="Year" value={result.release_year} />
                    <MetaRow label="Intent" value={result.intent?.replace(/_/g, " ")} />
                    <MetaRow label="Goal" value={result.goal} />
                  </div>
                  {result.content_themes.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-white/5">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                        Content Themes
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {result.content_themes.map((t) => (
                          <span
                            key={t}
                            className="text-xs bg-white/5 border border-white/10 text-gray-300 px-2 py-0.5 rounded"
                          >
                            {t.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </Card>
              </div>

              {/* Right: Awareness text */}
              <Card className="glass-panel p-6 border-white/10 flex flex-col">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                      Awareness Block
                    </h3>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs text-muted-foreground hover:text-white"
                    onClick={() => copy(result.awareness_text)}
                  >
                    <Copy className="w-3 h-3 mr-1" /> Copy
                  </Button>
                </div>

                <p className="text-xs text-muted-foreground mb-3">
                  This block is injected into the AI pipeline as context when
                  the URL is used as a topic.
                </p>

                <pre className="flex-1 text-sm text-gray-300 bg-black/40 rounded-xl p-4 whitespace-pre-wrap font-mono leading-relaxed border border-white/5 min-h-[200px]">
                  {result.awareness_text || (
                    <span className="text-muted-foreground italic">
                      No awareness signals extracted.
                    </span>
                  )}
                </pre>

                <div className="mt-4 pt-4 border-t border-white/5">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1.5">
                      <ExternalLink className="w-3 h-3" />
                      <a
                        href={result.canonical_url || result.raw_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-white transition-colors truncate max-w-[240px]"
                      >
                        {(result.canonical_url || result.raw_url).replace(
                          "https://",
                          "",
                        )}
                      </a>
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-xs text-muted-foreground hover:text-white"
                      onClick={() => copy(result.topic_string)}
                    >
                      <Copy className="w-3 h-3 mr-1" /> Copy topic
                    </Button>
                  </div>
                </div>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!result && !loading && !fetchError && (
        <div className="border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-muted-foreground min-h-[300px]">
          <Link2 className="w-12 h-12 mb-4 opacity-20" />
          <p>Paste any music or social URL above</p>
          <p className="text-xs mt-2 opacity-60">
            Spotify · TikTok · YouTube · Instagram · SoundCloud · and more
          </p>
        </div>
      )}
    </div>
  );
}
