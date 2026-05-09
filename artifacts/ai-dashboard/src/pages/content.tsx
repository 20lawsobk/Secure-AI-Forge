import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import {
  PenTool,
  Copy,
  Sparkles,
  History,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const BASE = "/api";

interface GenerationResult {
  platform: string;
  hook: string;
  body: string;
  cta: string;
  caption: string;
  hashtags: string[];
  processing_time_ms: number;
  source: string;
}

const PLATFORMS = [
  { value: "tiktok", label: "TikTok" },
  { value: "instagram", label: "Instagram Reels" },
  { value: "youtube", label: "YouTube Shorts" },
  { value: "twitter", label: "X (Twitter)" },
  { value: "linkedin", label: "LinkedIn" },
  { value: "facebook", label: "Facebook" },
];

const TONES = [
  { value: "energetic", label: "Energetic" },
  { value: "professional", label: "Professional" },
  { value: "casual", label: "Casual" },
  { value: "playful", label: "Playful" },
  { value: "edgy", label: "Edgy" },
];

const GOALS = [
  { value: "growth", label: "Growth & Viral" },
  { value: "conversion", label: "Conversion" },
  { value: "nurture", label: "Nurture & Educate" },
  { value: "engagement", label: "Engagement" },
];

export default function ContentGenerator() {
  const { toast } = useToast();
  const { adminKey } = useAuth();

  const [platform, setPlatform] = useState("tiktok");
  const [topic, setTopic] = useState("");
  const [tone, setTone] = useState("energetic");
  const [goal, setGoal] = useState("growth");
  const [history, setHistory] = useState<GenerationResult[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);

  const generateMut = useMutation({
    mutationFn: async (body: {
      platform: string;
      topic: string;
      tone: string;
      goal: string;
      include_hashtags: boolean;
    }): Promise<GenerationResult> => {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (adminKey) headers["X-Admin-Key"] = adminKey;
      const res = await fetch(`${BASE}/content/generate`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed: ${res.status}`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      setHistory((prev) => [data, ...prev].slice(0, 10));
      toast({ title: "Content generated!" });
    },
    onError: (e: Error) => {
      toast({
        title: "Generation failed",
        description: e.message,
        variant: "destructive",
      });
    },
  });

  const handleGenerate = () => {
    if (!topic.trim()) {
      toast({ title: "Please enter a topic", variant: "destructive" });
      return;
    }
    generateMut.mutate({
      platform,
      topic: topic.trim(),
      tone,
      goal,
      include_hashtags: true,
    });
  };

  const copyText = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: "Copied to clipboard" });
  };

  const result = generateMut.data;

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-10">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-fuchsia-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-primary/20">
          <Sparkles className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl font-display font-bold text-white mb-3">
          Content Generator
        </h1>
        <p className="text-muted-foreground text-lg">
          Harness the AI model to generate optimized scripts and captions across
          6 social platforms.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Controls */}
        <div className="lg:col-span-4 space-y-6">
          <Card className="glass-panel p-6 border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">
              Configuration
            </h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Platform
                </label>
                <Select value={platform} onValueChange={setPlatform}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {PLATFORMS.map((p) => (
                      <SelectItem key={p.value} value={p.value}>
                        {p.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Tone
                </label>
                <Select value={tone} onValueChange={setTone}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {TONES.map((t) => (
                      <SelectItem key={t.value} value={t.value}>
                        {t.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Goal
                </label>
                <Select value={goal} onValueChange={setGoal}>
                  <SelectTrigger className="bg-black/50 border-white/10 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-white/10 text-white">
                    {GOALS.map((g) => (
                      <SelectItem key={g.value} value={g.value}>
                        {g.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Topic / Idea
                </label>
                <Textarea
                  placeholder="e.g. 3 tips for starting a tech channel..."
                  className="bg-black/50 border-white/10 text-white min-h-[100px] resize-none"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && e.metaKey) handleGenerate();
                  }}
                />
                <p className="text-xs text-muted-foreground">⌘↵ to generate</p>
              </div>

              <Button
                className="w-full bg-gradient-to-r from-primary to-fuchsia-600 hover:from-primary/90 hover:to-fuchsia-600/90 text-white shadow-lg shadow-primary/25 h-12 text-base font-semibold mt-4"
                onClick={handleGenerate}
                disabled={generateMut.isPending}
              >
                {generateMut.isPending ? "Synthesizing..." : "Generate Content"}
              </Button>
            </div>
          </Card>

          {/* History */}
          {history.length > 0 && (
            <Card className="glass-panel border-white/10 overflow-hidden">
              <button
                className="w-full flex items-center justify-between p-4 text-sm font-medium text-white hover:bg-white/5 transition-colors"
                onClick={() => setHistoryOpen((o) => !o)}
              >
                <span className="flex items-center gap-2">
                  <History className="w-4 h-4 text-muted-foreground" />
                  Recent ({history.length})
                </span>
                {historyOpen ? (
                  <ChevronUp className="w-4 h-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                )}
              </button>
              <AnimatePresence>
                {historyOpen && (
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: "auto" }}
                    exit={{ height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 space-y-2 border-t border-white/5 pt-3 max-h-64 overflow-y-auto">
                      {history.map((h, i) => (
                        <div
                          key={i}
                          className="w-full text-left p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer"
                          onClick={() => copyText(h.caption)}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <Badge
                              variant="outline"
                              className="text-[10px] border-white/10 text-muted-foreground capitalize"
                            >
                              {h.platform}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {h.processing_time_ms.toFixed(0)}ms
                            </span>
                          </div>
                          <p className="text-xs text-gray-300 line-clamp-2">
                            {h.hook}
                          </p>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          )}
        </div>

        {/* Output */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {!result && !generateMut.isPending ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full min-h-[400px] border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-muted-foreground"
              >
                <PenTool className="w-12 h-12 mb-4 opacity-20" />
                <p>Configure parameters and click Generate</p>
                <p className="text-xs mt-2 opacity-60">
                  Or press ⌘↵ in the topic field
                </p>
              </motion.div>
            ) : generateMut.isPending ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full min-h-[400px] glass-panel rounded-2xl flex flex-col items-center justify-center text-primary"
              >
                <div className="w-10 h-10 border-4 border-primary/30 border-t-primary rounded-full animate-spin mb-4" />
                <p className="animate-pulse font-medium">
                  Model is thinking...
                </p>
              </motion.div>
            ) : (
              result && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel rounded-2xl p-6 space-y-6"
                >
                  <div className="flex justify-between items-center border-b border-white/10 pb-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="capitalize text-white font-medium bg-white/10 px-2 py-1 rounded">
                        {result.platform}
                      </span>
                      <span>•</span>
                      <span>{result.processing_time_ms.toFixed(0)}ms</span>
                      <span>•</span>
                      <span className="text-primary-foreground">
                        {result.source}
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyText(result.caption)}
                    >
                      <Copy className="w-4 h-4 mr-2" /> Copy Full
                    </Button>
                  </div>

                  <div className="space-y-5">
                    {[
                      {
                        label: "Hook",
                        value: result.hook,
                        labelColor: "text-primary",
                        isLarge: true,
                      },
                      {
                        label: "Body",
                        value: result.body,
                        labelColor: "text-muted-foreground",
                        isLarge: false,
                      },
                      {
                        label: "Call to Action",
                        value: result.cta,
                        labelColor: "text-amber-400",
                        isLarge: false,
                      },
                    ].map(({ label, value, labelColor, isLarge }) => (
                      <div key={label} className="group relative">
                        <div className="flex items-center justify-between mb-2">
                          <h4
                            className={`text-xs font-bold uppercase tracking-wider ${labelColor}`}
                          >
                            {label}
                          </h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground"
                            onClick={() => copyText(value)}
                          >
                            <Copy className="w-3 h-3 mr-1" /> Copy
                          </Button>
                        </div>
                        <p
                          className={`whitespace-pre-wrap leading-relaxed ${isLarge ? "text-lg text-white font-medium" : "text-gray-300"}`}
                        >
                          {value}
                        </p>
                      </div>
                    ))}
                  </div>

                  {result.hashtags.length > 0 && (
                    <div className="pt-4 border-t border-white/10">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-bold uppercase tracking-wider text-blue-400">
                          Hashtags
                        </h4>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs text-muted-foreground"
                          onClick={() => copyText(result.hashtags.join(" "))}
                        >
                          <Copy className="w-3 h-3 mr-1" /> Copy all
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {result.hashtags.map((tag: string) => (
                          <button
                            key={tag}
                            onClick={() => copyText(tag)}
                            className="text-sm text-blue-400 bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20 hover:bg-blue-500/20 transition-colors"
                          >
                            {tag}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
