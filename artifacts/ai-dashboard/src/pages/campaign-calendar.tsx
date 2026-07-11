import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import {
  CalendarDays,
  Sparkles,
  Save,
  Trash2,
  Send,
  Pencil,
  Clock,
  CheckCircle2,
  Circle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const BASE = "/api";

type PostStatus = "draft" | "scheduled" | "posted";

interface CampaignPost {
  post_id: string;
  phase: string;
  phase_label?: string;
  day_offset: number;
  date: string | null;
  platform: string;
  content_type: string;
  format: string;
  goal: string;
  brief?: string;
  hook: string;
  body: string;
  cta: string;
  caption: string;
  hashtags: string[];
  status: PostStatus;
  distribution?: { posting_time?: string | null; platform?: string } | null;
}

interface Campaign {
  campaign_id: string;
  name?: string;
  artist?: string;
  title?: string;
  genre?: string;
  release_date?: string | null;
  platforms?: string[];
  posts: CampaignPost[];
  calendar?: CampaignPost[];
  art_direction?: Record<string, unknown> | null;
}

interface CampaignSummary {
  campaign_id: string;
  name?: string;
  artist?: string;
  title?: string;
  release_date?: string | null;
  total_posts: number;
  by_status: Record<string, number>;
  first_date?: string | null;
  last_date?: string | null;
}

const PLATFORMS = [
  { value: "instagram", label: "Instagram" },
  { value: "tiktok", label: "TikTok" },
  { value: "youtube", label: "YouTube" },
  { value: "twitter", label: "X (Twitter)" },
  { value: "facebook", label: "Facebook" },
];

const STATUS_META: Record<
  PostStatus,
  { label: string; className: string; icon: typeof Circle }
> = {
  draft: {
    label: "Draft",
    className: "bg-white/5 text-muted-foreground border-white/10",
    icon: Circle,
  },
  scheduled: {
    label: "Scheduled",
    className: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    icon: Clock,
  },
  posted: {
    label: "Posted",
    className: "bg-green-500/15 text-green-300 border-green-500/30",
    icon: CheckCircle2,
  },
};

function fmtDate(d: string | null | undefined): string {
  if (!d) return "Unscheduled";
  try {
    return new Date(d + "T00:00:00").toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  } catch {
    return d;
  }
}

export default function CampaignCalendar() {
  const { toast } = useToast();
  const { adminKey, artistProfileId } = useAuth();
  const queryClient = useQueryClient();

  const [profileId, setProfileId] = useState(artistProfileId || "");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Generation form
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [genre, setGenre] = useState("");
  const [tone, setTone] = useState("energetic");
  const [weeks, setWeeks] = useState(6);
  const [releaseDate, setReleaseDate] = useState("");
  const [platforms, setPlatforms] = useState<string[]>(["instagram", "tiktok"]);

  const [editing, setEditing] = useState<CampaignPost | null>(null);

  const headers = useMemo(() => {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (adminKey) h["X-Admin-Key"] = adminKey;
    return h;
  }, [adminKey]);

  // ── Queries ────────────────────────────────────────────────────────────────
  const listQuery = useQuery({
    queryKey: ["campaigns", profileId],
    enabled: !!profileId,
    queryFn: async (): Promise<CampaignSummary[]> => {
      const res = await fetch(
        `${BASE}/campaigns?profile_id=${encodeURIComponent(profileId)}`,
        { headers },
      );
      if (!res.ok) throw new Error("Failed to load campaigns");
      return (await res.json()).campaigns ?? [];
    },
  });

  const detailQuery = useQuery({
    queryKey: ["campaign", profileId, selectedId],
    enabled: !!profileId && !!selectedId,
    queryFn: async (): Promise<Campaign> => {
      const res = await fetch(
        `${BASE}/campaigns/${selectedId}?profile_id=${encodeURIComponent(profileId)}`,
        { headers },
      );
      if (!res.ok) throw new Error("Failed to load campaign");
      return (await res.json()).campaign;
    },
  });

  // ── Mutations ────────────────────────────────────────────────────────────────
  const generateSaveMut = useMutation({
    mutationFn: async (): Promise<Campaign> => {
      if (!profileId.trim()) throw new Error("Set an artist profile id first");
      if (!title.trim()) throw new Error("A release title is required");
      const genRes = await fetch(`${BASE}/generate/campaign`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          title,
          artist_name: artist || undefined,
          genre: genre || undefined,
          tone: tone || undefined,
          weeks,
          release_date: releaseDate || undefined,
          platforms,
        }),
      });
      if (!genRes.ok) throw new Error("Campaign generation failed");
      const plan = await genRes.json();
      const saveRes = await fetch(`${BASE}/campaigns`, {
        method: "POST",
        headers,
        body: JSON.stringify({ profile_id: profileId, plan }),
      });
      if (!saveRes.ok) throw new Error("Saving the campaign failed");
      return (await saveRes.json()).campaign;
    },
    onSuccess: (c) => {
      toast({
        title: "Campaign saved",
        description: `${c.posts.length} posts added to the calendar.`,
      });
      setSelectedId(c.campaign_id);
      queryClient.invalidateQueries({ queryKey: ["campaigns", profileId] });
    },
    onError: (e: Error) =>
      toast({ title: "Could not build campaign", description: e.message, variant: "destructive" }),
  });

  const editPostMut = useMutation({
    mutationFn: async (patch: Partial<CampaignPost> & { post_id: string }) => {
      const res = await fetch(
        `${BASE}/campaigns/${selectedId}/posts/${patch.post_id}`,
        {
          method: "PATCH",
          headers,
          body: JSON.stringify({ profile_id: profileId, ...patch }),
        },
      );
      if (!res.ok) throw new Error("Failed to update post");
      return (await res.json()).post as CampaignPost;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["campaign", profileId, selectedId] });
      queryClient.invalidateQueries({ queryKey: ["campaigns", profileId] });
    },
    onError: (e: Error) =>
      toast({ title: "Update failed", description: e.message, variant: "destructive" }),
  });

  const scheduleAllMut = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${BASE}/campaigns/${selectedId}/schedule`, {
        method: "POST",
        headers,
        body: JSON.stringify({ profile_id: profileId, handoff: true }),
      });
      if (!res.ok) throw new Error("Scheduling hand-off failed");
      return await res.json();
    },
    onSuccess: (r) => {
      toast({
        title: "Handed off to distribution",
        description: `${r.scheduled_count} posts queued on their target dates.`,
      });
      queryClient.invalidateQueries({ queryKey: ["campaign", profileId, selectedId] });
      queryClient.invalidateQueries({ queryKey: ["campaigns", profileId] });
    },
    onError: (e: Error) =>
      toast({ title: "Hand-off failed", description: e.message, variant: "destructive" }),
  });

  const deleteMut = useMutation({
    mutationFn: async (id: string) => {
      const res = await fetch(
        `${BASE}/campaigns/${id}?profile_id=${encodeURIComponent(profileId)}`,
        { method: "DELETE", headers },
      );
      if (!res.ok) throw new Error("Delete failed");
    },
    onSuccess: (_d, id) => {
      if (selectedId === id) setSelectedId(null);
      queryClient.invalidateQueries({ queryKey: ["campaigns", profileId] });
      toast({ title: "Campaign deleted" });
    },
  });

  const campaign = detailQuery.data;
  const calendar = campaign?.calendar ?? campaign?.posts ?? [];

  function cyclePostStatus(p: CampaignPost) {
    const next: Record<PostStatus, PostStatus> = {
      draft: "scheduled",
      scheduled: "posted",
      posted: "draft",
    };
    editPostMut.mutate({ post_id: p.post_id, status: next[p.status] });
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-fuchsia-600 flex items-center justify-center shadow-lg shadow-primary/20">
          <CalendarDays className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="font-display font-bold text-2xl text-white">
            Campaign Calendar
          </h1>
          <p className="text-sm text-muted-foreground">
            Turn one release into a scheduled, editable rollout across your socials.
          </p>
        </div>
      </div>

      {/* Profile id */}
      <Card className="glass-panel p-4 border-white/10 flex flex-col sm:flex-row sm:items-end gap-4">
        <div className="flex-1">
          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
            Artist profile id (your calendar is saved under this)
          </label>
          <Input
            value={profileId}
            onChange={(e) => setProfileId(e.target.value)}
            placeholder="e.g. my-artist-name"
            className="bg-black/40 border-white/10 text-white"
          />
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: generator + saved list */}
        <div className="space-y-6 lg:col-span-1">
          <Card className="glass-panel p-5 border-white/10 space-y-4">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Sparkles className="w-4 h-4 text-primary" /> New rollout
            </div>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Release / song title *"
              className="bg-black/40 border-white/10 text-white"
            />
            <Input
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              placeholder="Artist name"
              className="bg-black/40 border-white/10 text-white"
            />
            <div className="grid grid-cols-2 gap-3">
              <Input
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                placeholder="Genre"
                className="bg-black/40 border-white/10 text-white"
              />
              <Select value={tone} onValueChange={setTone}>
                <SelectTrigger className="bg-black/40 border-white/10 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["energetic", "professional", "casual", "playful", "edgy"].map(
                    (t) => (
                      <SelectItem key={t} value={t}>
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                      </SelectItem>
                    ),
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">
                  Release date
                </label>
                <Input
                  type="date"
                  value={releaseDate}
                  onChange={(e) => setReleaseDate(e.target.value)}
                  className="bg-black/40 border-white/10 text-white"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">
                  Weeks ({weeks})
                </label>
                <Input
                  type="number"
                  min={2}
                  max={12}
                  value={weeks}
                  onChange={(e) => setWeeks(Number(e.target.value))}
                  className="bg-black/40 border-white/10 text-white"
                />
              </div>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">
                Platforms
              </label>
              <div className="flex flex-wrap gap-2">
                {PLATFORMS.map((p) => {
                  const on = platforms.includes(p.value);
                  return (
                    <button
                      key={p.value}
                      type="button"
                      onClick={() =>
                        setPlatforms((prev) =>
                          on
                            ? prev.filter((x) => x !== p.value)
                            : [...prev, p.value],
                        )
                      }
                      className={`px-3 py-1 rounded-full text-xs border transition-colors ${
                        on
                          ? "bg-primary/15 text-primary border-primary/40"
                          : "bg-white/5 text-muted-foreground border-white/10 hover:text-white"
                      }`}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <Button
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
              disabled={generateSaveMut.isPending}
              onClick={() => generateSaveMut.mutate()}
            >
              {generateSaveMut.isPending ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Save className="w-4 h-4 mr-2" />
              )}
              Generate & save calendar
            </Button>
          </Card>

          {/* Saved campaigns */}
          <Card className="glass-panel p-5 border-white/10 space-y-3">
            <div className="text-white font-semibold text-sm">
              Saved campaigns
            </div>
            {!profileId && (
              <p className="text-xs text-muted-foreground">
                Enter an artist profile id above to see saved campaigns.
              </p>
            )}
            {profileId && listQuery.isLoading && (
              <p className="text-xs text-muted-foreground">Loading…</p>
            )}
            {profileId &&
              !listQuery.isLoading &&
              (listQuery.data?.length ?? 0) === 0 && (
                <p className="text-xs text-muted-foreground">
                  No saved campaigns yet.
                </p>
              )}
            <div className="space-y-2">
              {listQuery.data?.map((c) => (
                <button
                  key={c.campaign_id}
                  onClick={() => setSelectedId(c.campaign_id)}
                  className={`w-full text-left p-3 rounded-xl border transition-colors ${
                    selectedId === c.campaign_id
                      ? "bg-primary/10 border-primary/40"
                      : "bg-white/5 border-white/10 hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-white truncate">
                      {c.name || c.title || "Untitled"}
                    </span>
                    <Trash2
                      className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive flex-shrink-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteMut.mutate(c.campaign_id);
                      }}
                    />
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-1">
                    {c.total_posts} posts · {c.by_status?.scheduled ?? 0}{" "}
                    scheduled · {c.by_status?.posted ?? 0} posted
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* Right: calendar */}
        <div className="lg:col-span-2">
          {!selectedId && (
            <Card className="glass-panel p-10 border-white/10 text-center text-muted-foreground">
              Select or generate a campaign to see its calendar.
            </Card>
          )}
          {selectedId && detailQuery.isLoading && (
            <Card className="glass-panel p-10 border-white/10 text-center text-muted-foreground">
              <Loader2 className="w-5 h-5 animate-spin mx-auto" />
            </Card>
          )}
          {campaign && (
            <div className="space-y-4">
              <Card className="glass-panel p-5 border-white/10 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-white font-semibold">
                    {campaign.title}
                    {campaign.artist ? (
                      <span className="text-muted-foreground font-normal">
                        {" "}
                        · {campaign.artist}
                      </span>
                    ) : null}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    Release {fmtDate(campaign.release_date)} ·{" "}
                    {calendar.length} posts
                  </div>
                </div>
                <Button
                  size="sm"
                  className="bg-primary hover:bg-primary/90 text-primary-foreground"
                  disabled={scheduleAllMut.isPending}
                  onClick={() => scheduleAllMut.mutate()}
                >
                  {scheduleAllMut.isPending ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4 mr-2" />
                  )}
                  Queue all to distribution
                </Button>
              </Card>

              <div className="space-y-3">
                {calendar.map((p) => {
                  const meta = STATUS_META[p.status] ?? STATUS_META.draft;
                  const StatusIcon = meta.icon;
                  return (
                    <motion.div
                      key={p.post_id}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <Card className="glass-panel p-4 border-white/10">
                        <div className="flex items-start gap-4">
                          {/* Date rail */}
                          <div className="flex-shrink-0 w-20 text-center">
                            <div className="text-xs font-semibold text-white">
                              {fmtDate(p.date)}
                            </div>
                            <div className="text-[10px] text-muted-foreground mt-0.5">
                              {p.day_offset === 0
                                ? "Release day"
                                : p.day_offset > 0
                                  ? `+${p.day_offset}d`
                                  : `${p.day_offset}d`}
                            </div>
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex flex-wrap items-center gap-1.5 mb-1.5">
                              <Badge className="bg-white/5 text-muted-foreground border-white/10 capitalize">
                                {p.platform}
                              </Badge>
                              <Badge className="bg-white/5 text-muted-foreground border-white/10 capitalize">
                                {p.format}
                              </Badge>
                              <Badge className="bg-fuchsia-500/10 text-fuchsia-300 border-fuchsia-500/20 capitalize">
                                {p.phase_label || p.phase}
                              </Badge>
                              <button
                                onClick={() => cyclePostStatus(p)}
                                title="Click to change status"
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] border ${meta.className}`}
                              >
                                <StatusIcon className="w-3 h-3" />
                                {meta.label}
                              </button>
                            </div>
                            <div className="text-sm font-medium text-white">
                              {p.hook}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap line-clamp-3">
                              {p.body}
                            </div>
                            {p.distribution?.posting_time && (
                              <div className="text-[11px] text-amber-300/80 mt-1.5">
                                Queued · best time {p.distribution.posting_time}
                              </div>
                            )}
                          </div>

                          <Button
                            size="icon"
                            variant="ghost"
                            className="text-muted-foreground hover:text-white flex-shrink-0"
                            onClick={() => setEditing(p)}
                          >
                            <Pencil className="w-4 h-4" />
                          </Button>
                        </div>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Edit dialog */}
      <EditPostDialog
        post={editing}
        onClose={() => setEditing(null)}
        saving={editPostMut.isPending}
        onSave={(patch) => {
          editPostMut.mutate(
            { post_id: editing!.post_id, ...patch },
            { onSuccess: () => setEditing(null) },
          );
        }}
      />
    </div>
  );
}

function EditPostDialog({
  post,
  onClose,
  onSave,
  saving,
}: {
  post: CampaignPost | null;
  onClose: () => void;
  onSave: (patch: Partial<CampaignPost>) => void;
  saving: boolean;
}) {
  const [hook, setHook] = useState("");
  const [body, setBody] = useState("");
  const [cta, setCta] = useState("");
  const [date, setDate] = useState("");
  const [platform, setPlatform] = useState("instagram");
  const [status, setStatus] = useState<PostStatus>("draft");

  // Sync local state when a new post opens.
  useEffect(() => {
    if (post) {
      setHook(post.hook);
      setBody(post.body);
      setCta(post.cta);
      setDate(post.date ?? "");
      setPlatform(post.platform);
      setStatus(post.status);
    }
  }, [post]);

  return (
    <Dialog open={!!post} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="glass-panel border-white/10 sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-white">Edit post</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 py-2">
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Hook</label>
            <Input
              value={hook}
              onChange={(e) => setHook(e.target.value)}
              className="bg-black/40 border-white/10 text-white"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Body</label>
            <Textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              rows={4}
              className="bg-black/40 border-white/10 text-white"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">CTA</label>
            <Input
              value={cta}
              onChange={(e) => setCta(e.target.value)}
              className="bg-black/40 border-white/10 text-white"
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Date</label>
              <Input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="bg-black/40 border-white/10 text-white"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                Platform
              </label>
              <Select value={platform} onValueChange={setPlatform}>
                <SelectTrigger className="bg-black/40 border-white/10 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PLATFORMS.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                Status
              </label>
              <Select
                value={status}
                onValueChange={(v) => setStatus(v as PostStatus)}
              >
                <SelectTrigger className="bg-black/40 border-white/10 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="draft">Draft</SelectItem>
                  <SelectItem value="scheduled">Scheduled</SelectItem>
                  <SelectItem value="posted">Posted</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            className="bg-primary hover:bg-primary/90 text-primary-foreground"
            disabled={saving}
            onClick={() =>
              onSave({ hook, body, cta, date: date || undefined, platform, status })
            }
          >
            {saving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            Save changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
