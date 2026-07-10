import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Mic2,
  Sparkles,
  Palette,
  ShieldAlert,
  Save,
  Loader2,
  Eye,
} from "lucide-react";
import { useAuth, getAuthHeaders } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";

const BASE = "/api";

// Mirrors ai_model/request_intelligence.apply_disclosure() in the training
// server — kept in sync manually since disclosure text is a fixed literal
// there, not something the API returns for previewing.
const AI_DISCLOSURE_LABEL = "✨ Crafted with AI-assisted creative tools.";

const SAMPLE_CAPTION =
  "Been living in the studio all week — new single drops Friday. Can't wait for you to hear this one.";

interface ArtistProfile {
  artist_name?: string;
  current_single?: string;
  current_album?: string;
  audience_age?: string;
  audience_geo?: string;
  genre?: string;
  tone?: string;
  vocabulary?: string[];
  avoid_words?: string[];
  palette?: string[];
  ai_disclosure?: boolean;
  updated_at?: number;
}

const profileSchema = z.object({
  artist_name: z.string().optional(),
  current_single: z.string().optional(),
  audience_age: z.string().optional(),
  audience_geo: z.string().optional(),
  genre: z.string().optional(),
  tone: z.string().optional(),
  vocabulary: z.string().optional(),
  avoid_words: z.string().optional(),
  palette: z.string().optional(),
  ai_disclosure: z.boolean(),
});

type ProfileFormValues = z.infer<typeof profileSchema>;

// "sunrise orange, deep navy, #ffffff" -> ["sunrise orange", "deep navy", "#ffffff"]
function toList(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
}

function fromList(value: string[] | undefined): string {
  return (value ?? []).join(", ");
}

const HEX_RE = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i;

function isHex(swatch: string): boolean {
  return HEX_RE.test(swatch.trim());
}

export default function ArtistSettings() {
  const { adminKey, artistProfileId, setArtistProfileId } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [profileIdInput, setProfileIdInput] = useState(artistProfileId);

  const activeId = artistProfileId.trim();

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["artist-profile", activeId, adminKey],
    queryFn: async () => {
      const res = await fetch(
        `${BASE}/storage/artist/${encodeURIComponent(activeId)}`,
        { headers: getAuthHeaders() as Record<string, string> },
      );
      if (!res.ok) throw new Error("Failed to load profile");
      return res.json() as Promise<{
        profile_id: string;
        profile: ArtistProfile;
        releases: unknown[];
      }>;
    },
    enabled: !!adminKey && !!activeId,
    retry: false,
  });

  const form = useForm<ProfileFormValues>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      artist_name: "",
      current_single: "",
      audience_age: "",
      audience_geo: "",
      genre: "",
      tone: "",
      vocabulary: "",
      avoid_words: "",
      palette: "",
      ai_disclosure: false,
    },
  });

  // Populate the form whenever a saved profile loads for the active id.
  useEffect(() => {
    const p = data?.profile;
    if (!p) return;
    form.reset({
      artist_name: p.artist_name ?? "",
      current_single: p.current_single ?? "",
      audience_age: p.audience_age ?? "",
      audience_geo: p.audience_geo ?? "",
      genre: p.genre ?? "",
      tone: p.tone ?? "",
      vocabulary: fromList(p.vocabulary),
      avoid_words: fromList(p.avoid_words),
      palette: fromList(p.palette),
      ai_disclosure: !!p.ai_disclosure,
    });
  }, [data, form]);

  const saveMut = useMutation({
    mutationFn: async (values: ProfileFormValues) => {
      // The backend merge-updates by field (storage_client.save_profile):
      // any field that is `None` is left untouched rather than cleared. Send
      // "" (not null) for blanked text fields so clearing a field in the UI
      // actually clears it in storage instead of silently no-op'ing.
      const body = {
        artist_name: values.artist_name ?? "",
        current_single: values.current_single ?? "",
        audience_age: values.audience_age ?? "",
        audience_geo: values.audience_geo ?? "",
        genre: values.genre ?? "",
        tone: values.tone ?? "",
        vocabulary: toList(values.vocabulary),
        avoid_words: toList(values.avoid_words),
        palette: toList(values.palette),
        ai_disclosure: values.ai_disclosure,
      };
      const res = await fetch(
        `${BASE}/storage/artist/${encodeURIComponent(activeId)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...getAuthHeaders(),
          } as Record<string, string>,
          body: JSON.stringify(body),
        },
      );
      if (!res.ok) throw new Error("Failed to save profile");
      return res.json();
    },
    onSuccess: () => {
      toast({ title: "Brand Voice profile saved" });
      queryClient.invalidateQueries({ queryKey: ["artist-profile", activeId] });
    },
    onError: () => {
      toast({ variant: "destructive", title: "Failed to save profile" });
    },
  });

  const onSubmit = (values: ProfileFormValues) => saveMut.mutate(values);

  const watched = form.watch();
  const disclosureOn = watched.ai_disclosure;
  const previewCaption = disclosureOn
    ? `${SAMPLE_CAPTION}\n\n${AI_DISCLOSURE_LABEL}`
    : SAMPLE_CAPTION;
  const paletteSwatches = toList(watched.palette);

  if (!adminKey) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <div className="w-16 h-16 rounded-2xl bg-destructive/10 flex items-center justify-center mb-4">
          <ShieldAlert className="w-8 h-8 text-destructive" />
        </div>
        <h2 className="text-2xl font-display font-bold text-white mb-2">
          Authentication Required
        </h2>
        <p className="text-muted-foreground max-w-md">
          You need to provide an Admin Key in the sidebar to view and manage a
          Brand Voice profile.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-3xl font-display font-bold text-white flex items-center gap-3">
          <Mic2 className="w-7 h-7 text-primary" /> Brand Voice
        </h1>
        <p className="text-muted-foreground mt-1">
          Set the tone, genre, and vocabulary the AI should reuse in every
          generated caption, image, and video for this artist — and control
          whether generated content discloses AI assistance.
        </p>
      </div>

      <Card className="glass-panel border-white/10 p-6 space-y-2">
        <Label className="text-white">Artist profile ID</Label>
        <p className="text-xs text-muted-foreground">
          Identifies which artist's Brand Voice profile to load and save.
          Reuse the same ID everywhere you generate content for this artist.
        </p>
        <div className="flex gap-2 max-w-md">
          <Input
            value={profileIdInput}
            onChange={(e) => setProfileIdInput(e.target.value)}
            placeholder="e.g. jane-doe-music"
            className="bg-black/30 border-white/10 text-white"
          />
          <Button
            variant="secondary"
            onClick={() => setArtistProfileId(profileIdInput.trim())}
            disabled={!profileIdInput.trim() || profileIdInput.trim() === activeId}
          >
            Load
          </Button>
        </div>
      </Card>

      {!activeId ? (
        <div className="glass-panel rounded-2xl p-10 text-center text-muted-foreground">
          Enter an artist profile ID above and click Load to view or create a
          Brand Voice profile.
        </div>
      ) : isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full rounded-2xl bg-white/5" />
          <Skeleton className="h-40 w-full rounded-2xl bg-white/5" />
        </div>
      ) : (
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <Card className="glass-panel border-white/10 p-6 space-y-4">
              <h3 className="text-sm font-medium text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" /> Identity
                {isFetching && (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
                )}
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="artist_name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">Artist name</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="Jane Doe"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="current_single"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">
                        Current single / release
                      </FormLabel>
                      <FormControl>
                        <Input
                          placeholder="Midnight Drive"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="audience_age"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">
                        Audience age range
                      </FormLabel>
                      <FormControl>
                        <Input
                          placeholder="18-24"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="audience_geo"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">
                        Audience geography
                      </FormLabel>
                      <FormControl>
                        <Input
                          placeholder="US, UK, Brazil"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
              </div>
            </Card>

            <Card className="glass-panel border-white/10 p-6 space-y-4">
              <h3 className="text-sm font-medium text-white flex items-center gap-2">
                <Mic2 className="w-4 h-4 text-primary" /> Voice
              </h3>
              <p className="text-xs text-muted-foreground">
                Used as the default genre/tone whenever a generation request
                doesn't specify one — a per-request value always wins.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="genre"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">Genre</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="lofi hip-hop"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="tone"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-white">Tone</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="warm, intimate, a little cheeky"
                          className="bg-black/30 border-white/10 text-white"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
              </div>
              <FormField
                control={form.control}
                name="vocabulary"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-white">
                      Favored vocabulary
                    </FormLabel>
                    <FormDescription className="text-xs">
                      Comma-separated words/phrases generation should favor.
                    </FormDescription>
                    <FormControl>
                      <Textarea
                        placeholder="fam, no cap, studio nights"
                        className="bg-black/30 border-white/10 text-white min-h-[70px]"
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="avoid_words"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-white">
                      Words to avoid
                    </FormLabel>
                    <FormDescription className="text-xs">
                      Comma-separated words/phrases generation should never
                      use.
                    </FormDescription>
                    <FormControl>
                      <Textarea
                        placeholder="cringe, lit, epic"
                        className="bg-black/30 border-white/10 text-white min-h-[70px]"
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </Card>

            <Card className="glass-panel border-white/10 p-6 space-y-4">
              <h3 className="text-sm font-medium text-white flex items-center gap-2">
                <Palette className="w-4 h-4 text-primary" /> Visual palette
              </h3>
              <FormField
                control={form.control}
                name="palette"
                render={({ field }) => (
                  <FormItem>
                    <FormDescription className="text-xs">
                      Comma-separated hex colors (or color names) used to bias
                      generated image style.
                    </FormDescription>
                    <FormControl>
                      <Input
                        placeholder="#1a1a2e, #e94560, midnight blue"
                        className="bg-black/30 border-white/10 text-white"
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              {paletteSwatches.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {paletteSwatches.map((swatch, i) =>
                    isHex(swatch) ? (
                      <div
                        key={`${swatch}-${i}`}
                        className="w-8 h-8 rounded-full border border-white/20"
                        style={{ backgroundColor: swatch }}
                        title={swatch}
                      />
                    ) : (
                      <Badge
                        key={`${swatch}-${i}`}
                        variant="secondary"
                        className="bg-white/5 text-xs font-normal border-white/10"
                      >
                        {swatch}
                      </Badge>
                    ),
                  )}
                </div>
              )}
            </Card>

            <Card className="glass-panel border-white/10 p-6 space-y-4">
              <h3 className="text-sm font-medium text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-primary" /> AI disclosure &
                preview
              </h3>
              <FormField
                control={form.control}
                name="ai_disclosure"
                render={({ field }) => (
                  <FormItem className="flex items-center justify-between rounded-xl border border-white/10 bg-black/20 p-4">
                    <div className="space-y-0.5">
                      <FormLabel className="text-white">
                        Disclose AI-assisted content
                      </FormLabel>
                      <FormDescription className="text-xs">
                        When on, generated captions add a short "Crafted with
                        AI-assisted creative tools" line.
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <div className="rounded-xl border border-white/10 bg-black/30 p-4">
                <p className="text-xs text-muted-foreground mb-2">
                  Preview — how a generated caption will look
                </p>
                <p className="text-white whitespace-pre-line text-sm">
                  {previewCaption}
                </p>
              </div>
            </Card>

            <div className="flex justify-end">
              <Button
                type="submit"
                disabled={saveMut.isPending}
                className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20"
              >
                {saveMut.isPending ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Save className="w-4 h-4 mr-2" />
                )}
                Save Brand Voice
              </Button>
            </div>
          </form>
        </Form>
      )}
    </div>
  );
}
