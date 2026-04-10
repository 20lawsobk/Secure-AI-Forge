import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import { PenTool, Check, Copy, Sparkles, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const BASE = "/api";

export default function ContentGenerator() {
  const { toast } = useToast();
  const { adminKey } = useAuth();

  const generateMut = useMutation({
    mutationFn: async (body: {
      platform: string;
      topic: string;
      tone: string;
      goal: string;
      include_hashtags: boolean;
    }) => {
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
  });

  const [platform, setPlatform] = useState("tiktok");
  const [topic, setTopic] = useState("");
  const [tone, setTone] = useState("energetic");
  const [goal, setGoal] = useState("growth");

  const handleGenerate = async () => {
    if (!topic) {
      toast({ title: "Please enter a topic", variant: "destructive" });
      return;
    }

    try {
      await generateMut.mutateAsync({
        platform,
        topic,
        tone,
        goal,
        include_hashtags: true,
      });
      toast({ title: "Content generated!" });
    } catch (e: any) {
      toast({
        title: "Generation failed",
        description: e?.message,
        variant: "destructive",
      });
    }
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
          Harness the AI model to generate highly optimized scripts and captions
          across 8 social platforms.
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
                    <SelectItem value="tiktok">TikTok</SelectItem>
                    <SelectItem value="instagram">Instagram Reels</SelectItem>
                    <SelectItem value="youtube">YouTube Shorts</SelectItem>
                    <SelectItem value="twitter">X (Twitter)</SelectItem>
                    <SelectItem value="linkedin">LinkedIn</SelectItem>
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
                    <SelectItem value="energetic">Energetic</SelectItem>
                    <SelectItem value="professional">Professional</SelectItem>
                    <SelectItem value="casual">Casual</SelectItem>
                    <SelectItem value="playful">Playful</SelectItem>
                    <SelectItem value="edgy">Edgy</SelectItem>
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
                    <SelectItem value="growth">Growth & Viral</SelectItem>
                    <SelectItem value="conversion">Conversion</SelectItem>
                    <SelectItem value="nurture">Nurture & Educate</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium text-muted-foreground">
                  Topic / Idea
                </label>
                <Textarea
                  placeholder="e.g., 3 tips for starting a tech channel..."
                  className="bg-black/50 border-white/10 text-white min-h-[100px] resize-none"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
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
        </div>

        {/* Output */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {!result && !generateMut.isPending ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full min-h-[400px] border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center text-muted-foreground"
              >
                <PenTool className="w-12 h-12 mb-4 opacity-20" />
                <p>Configure parameters and click Generate</p>
              </motion.div>
            ) : generateMut.isPending ? (
              <motion.div
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

                  <div className="space-y-4">
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-primary mb-2">
                        Hook
                      </h4>
                      <p className="text-lg text-white font-medium">
                        {result.hook}
                      </p>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">
                        Body
                      </h4>
                      <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                        {result.body}
                      </p>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-amber-400 mb-2">
                        Call to Action
                      </h4>
                      <p className="text-white font-medium">{result.cta}</p>
                    </div>
                  </div>

                  {result.hashtags.length > 0 && (
                    <div className="pt-4 border-t border-white/10">
                      <div className="flex flex-wrap gap-2">
                        {result.hashtags.map((tag: string) => (
                          <span
                            key={tag}
                            className="text-sm text-blue-400 bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20"
                          >
                            {tag}
                          </span>
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
