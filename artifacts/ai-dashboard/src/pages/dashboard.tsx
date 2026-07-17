import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  useGetDashboardStats,
  useHealthCheck,
} from "@workspace/api-client-react";
import { StatCard } from "@/components/stat-card";
import { useAuth } from "@/hooks/use-auth";
import {
  Server,
  KeySquare,
  Cpu,
  Database,
  Activity,
  FileJson,
  Zap,
  HardDrive,
  ShieldCheck,
  AlertTriangle,
  Clock,
  RefreshCw,
  Music,
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatUptime } from "@/lib/format";

const BASE = "/api";

export default function Dashboard() {
  const { adminKey } = useAuth();
  const authHdr: Record<string, string> = adminKey
    ? { "X-Admin-Key": adminKey }
    : {};

  const {
    data: stats,
    isLoading: statsLoading,
    refetch: refetchStats,
    isFetching: statsFetching,
  } = useGetDashboardStats({
    query: { refetchInterval: 15_000 },
  });

  const {
    data: health,
    isLoading: healthLoading,
    refetch: refetchHealth,
  } = useHealthCheck({
    query: { refetchInterval: 20_000 },
  });

  const { data: storageStatus, refetch: refetchStorage } = useQuery({
    queryKey: ["storage-status", adminKey],
    queryFn: async () => {
      const res = await fetch(`${BASE}/storage/status`, { headers: authHdr });
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    enabled: !!adminKey,
    refetchInterval: 20_000,
    retry: false,
  });

  const { data: watchdog, refetch: refetchWatchdog } = useQuery({
    queryKey: ["watchdog-status", adminKey],
    queryFn: async () => {
      const res = await fetch(`${BASE}/watchdog/status`, { headers: authHdr });
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    enabled: !!adminKey,
    refetchInterval: 25_000,
    retry: false,
  });

  const { data: audioStatus, refetch: refetchAudio } = useQuery({
    queryKey: ["audio-dataset-status", adminKey],
    queryFn: async () => {
      const res = await fetch(`${BASE}/storage/datasets/audio/status`, {
        headers: authHdr,
      });
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    enabled: !!adminKey,
    refetchInterval: 30_000,
    retry: false,
  });

  const [seeding, setSeeding] = useState(false);

  const handleSeedAudio = async () => {
    if (seeding || audioStatus?.seeding_now) return;
    setSeeding(true);
    try {
      await fetch(`${BASE}/storage/datasets/audio/seed`, {
        method: "POST",
        headers: authHdr,
      });
      // Poll for status update
      setTimeout(() => refetchAudio(), 2_000);
      setTimeout(() => refetchAudio(), 8_000);
    } catch (_) {
      // non-fatal — status panel will reflect actual state
    } finally {
      setSeeding(false);
    }
  };

  const handleRefreshAll = () => {
    refetchStats();
    refetchHealth();
    if (adminKey) {
      refetchStorage();
      refetchWatchdog();
      refetchAudio();
    }
  };

  const uptimeFormatted = formatUptime(health?.uptime_seconds);
  // storage_mode is the canonical field added in the /storage/status response.
  // Fall back to deriving from `available` so old responses still work.
  const storageMode: "live" | "local_fallback" | "offline" | "unknown" =
    storageStatus?.storage_mode ??
    (storageStatus?.available === true
      ? "live"
      : storageStatus?.disk_store_available
        ? "local_fallback"
        : storageStatus
          ? "offline"
          : "unknown");
  const storageOnline = storageMode === "live";
  const storageDegraded = storageMode === "local_fallback";
  const storageOffline = storageMode === "offline";
  const watchdogAlerts: string[] = watchdog?.active_alerts ?? [];

  if (statsLoading || healthLoading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Skeleton className="h-10 w-48 bg-white/5" />
          <Skeleton className="h-8 w-24 bg-white/5" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-40 w-full rounded-2xl bg-white/5" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
        <div>
          <h1 className="text-3xl font-display font-bold text-white">
            System Overview
          </h1>
          <p className="text-muted-foreground mt-1">
            Live metrics from the MaxCore AI cluster.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefreshAll}
            disabled={statsFetching}
            className="text-muted-foreground hover:text-white h-8 px-3"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 mr-1.5 ${statsFetching ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>

          <div className="glass-panel px-4 py-2 rounded-xl flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className={`w-2.5 h-2.5 rounded-full ${health?.status === "healthy" ? "bg-green-500 animate-pulse" : "bg-destructive"}`}
              />
              <span className="text-sm font-medium text-white capitalize">
                {health?.status || "Unknown"}
              </span>
            </div>
            <div className="w-px h-4 bg-border" />
            <span className="text-xs text-muted-foreground font-mono">
              v{health?.version || "1.0.0"}
            </span>
            {uptimeFormatted && uptimeFormatted !== "—" && (
              <>
                <div className="w-px h-4 bg-border" />
                <span className="text-xs text-muted-foreground font-mono flex items-center gap-1">
                  <Clock className="w-3 h-3" /> {uptimeFormatted}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Storage degraded / offline banner */}
      {(storageDegraded || storageOffline) && (
        <div className={`glass-panel p-4 rounded-xl flex items-start gap-3 ${storageOffline ? "border border-red-500/30 bg-red-500/5" : "border border-amber-500/30 bg-amber-500/5"}`}>
          <AlertTriangle className={`w-5 h-5 mt-0.5 shrink-0 ${storageOffline ? "text-red-400" : "text-amber-400"}`} />
          <div className="space-y-1">
            <p className={`text-sm font-medium ${storageOffline ? "text-red-300" : "text-amber-300"}`}>
              pdim Storage {storageOffline ? "Offline" : "Degraded — Local Fallback Active"}
            </p>
            <p className={`text-xs ${storageOffline ? "text-red-400/80" : "text-amber-400/80"}`}>
              {storageOffline
                ? "Neither pdim nor the local disk store is reachable. Content generation quality is significantly reduced."
                : "pdim is unreachable. The system is using the local disk-backed store. Quality awareness and retrieval index are running in fallback mode — output quality may be reduced."}
            </p>
          </div>
        </div>
      )}

      {/* Watchdog alerts banner */}
      {watchdogAlerts.length > 0 && (
        <div className="glass-panel p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="text-sm font-medium text-amber-300">
              Watchdog Alerts ({watchdogAlerts.length})
            </p>
            {watchdogAlerts.map((alert: string, i: number) => (
              <p key={i} className="text-xs text-amber-400/80">
                • {alert}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Main Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Active API Keys"
          value={stats?.active_api_keys ?? 0}
          icon={KeySquare}
          description={`Out of ${stats?.total_api_keys ?? 0} total keys`}
        />
        <StatCard
          title="Requests Today"
          value={(stats?.total_requests_today ?? 0).toLocaleString()}
          icon={Activity}
          description="Across all endpoints"
        />
        <StatCard
          title="GPU Lanes"
          value={stats?.gpu_lanes ?? 0}
          icon={Cpu}
          description="HyperGPU Cluster"
        />
        <StatCard
          title="Vocab Size"
          value={(stats?.vocab_size ?? 0).toLocaleString()}
          icon={Database}
          description="Tokens in vocabulary"
        />
      </div>

      {/* Secondary Status Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Engine */}
        <div className="glass-panel p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Server className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">
              Model Engine
            </h3>
          </div>

          <div className="space-y-3">
            {[
              {
                label: "Engine Status",
                content: (
                  <Badge
                    variant="outline"
                    className={
                      stats?.model_status === "running"
                        ? "border-green-500/50 text-green-400"
                        : ""
                    }
                  >
                    {stats?.model_status || "Unknown"}
                  </Badge>
                ),
              },
              {
                label: "Weights Loaded",
                content: (
                  <Badge
                    variant="outline"
                    className={
                      stats?.weights_exist
                        ? "border-primary/50 text-primary"
                        : "border-destructive/50 text-destructive"
                    }
                  >
                    {stats?.weights_exist ? "Yes" : "No"}
                  </Badge>
                ),
              },
              {
                label: "Training State",
                content: (
                  <div className="flex items-center gap-2">
                    {stats?.training_state === "running" && (
                      <Zap className="w-4 h-4 text-amber-400" />
                    )}
                    <span className="text-sm font-medium text-white capitalize">
                      {stats?.training_state || "Idle"}
                    </span>
                  </div>
                ),
              },
            ].map(({ label, content }) => (
              <div
                key={label}
                className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5"
              >
                <span className="text-sm text-muted-foreground">{label}</span>
                {content}
              </div>
            ))}
          </div>
        </div>

        {/* Storage Status */}
        <div className="glass-panel p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-lg">
              <HardDrive className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">
              pdim Storage
            </h3>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
              <span className="text-sm text-muted-foreground">Connection</span>
              <Badge
                variant="outline"
                className={
                  storageOnline
                    ? "border-green-500/50 text-green-400"
                    : storageDegraded
                      ? "border-amber-500/50 text-amber-400"
                      : storageOffline
                        ? "border-red-500/50 text-red-400"
                        : "border-white/20 text-muted-foreground"
                }
              >
                {storageOnline
                  ? "Online"
                  : storageDegraded
                    ? "Degraded"
                    : storageOffline
                      ? "Offline"
                      : storageStatus
                        ? "Unknown"
                        : "Checking…"}
              </Badge>
            </div>
            {storageStatus && (
              <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                <span className="text-sm text-muted-foreground">Mode</span>
                <span
                  className={`text-xs font-mono ${
                    storageOnline
                      ? "text-green-400"
                      : storageDegraded
                        ? "text-amber-400"
                        : storageOffline
                          ? "text-red-400"
                          : "text-muted-foreground"
                  }`}
                >
                  {storageMode === "live"
                    ? "live"
                    : storageMode === "local_fallback"
                      ? "local fallback"
                      : storageMode === "offline"
                        ? "offline"
                        : "unknown"}
                </span>
              </div>
            )}
            {(storageStatus?.instance_id ?? storageStatus?.instance) && (
              <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                <span className="text-sm text-muted-foreground">Instance</span>
                <span className="text-xs font-mono text-white truncate max-w-[120px]">
                  {storageStatus.instance_id ?? storageStatus.instance}
                </span>
              </div>
            )}
            {(storageStatus?.keys_count ?? storageStatus?.fallback_keys) != null && (
              <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                <span className="text-sm text-muted-foreground">
                  {storageOnline ? "Keys stored" : "Fallback keys"}
                </span>
                <span className="text-sm font-mono text-white">
                  {storageStatus.keys_count ?? storageStatus.fallback_keys}
                </span>
              </div>
            )}
            {!storageStatus && (
              <div className="flex items-center justify-center h-[72px] bg-black/20 rounded-xl border border-white/5 border-dashed">
                <button
                  onClick={() =>
                    window.dispatchEvent(new Event("openAuthDialog"))
                  }
                  className="text-xs text-primary hover:text-primary/80 transition-colors underline underline-offset-2 cursor-pointer"
                >
                  Enter admin key
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Knowledge Base + Watchdog */}
        <div className="space-y-4">
          <div className="glass-panel p-6 rounded-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-primary/10 rounded-lg">
                <FileJson className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-lg font-display font-semibold text-white">
                Knowledge Base
              </h3>
            </div>

            <div className="flex flex-col items-center justify-center h-[96px] bg-black/20 rounded-xl border border-white/5 border-dashed">
              <p className="text-4xl font-display font-bold text-white mb-1">
                {stats?.boostsheet_count || 0}
              </p>
              <p className="text-sm text-muted-foreground">Total BoostSheets</p>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-2xl">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <ShieldCheck className="w-4 h-4 text-primary" />
              </div>
              <h3 className="text-base font-display font-semibold text-white">
                Watchdog
              </h3>
              <Badge
                variant="outline"
                className={
                  watchdog?.status === "healthy"
                    ? "ml-auto border-green-500/40 text-green-400"
                    : watchdogAlerts.length > 0
                      ? "ml-auto border-amber-500/40 text-amber-400"
                      : "ml-auto border-white/10 text-muted-foreground"
                }
              >
                {watchdog?.status || "Unknown"}
              </Badge>
            </div>
            {watchdog ? (
              <div className="space-y-1.5 text-xs">
                {watchdog.checks_run != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Checks run</span>
                    <span className="text-white font-mono">
                      {watchdog.checks_run}
                    </span>
                  </div>
                )}
                {watchdog.restarts != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Auto-restarts</span>
                    <span className="text-white font-mono">
                      {watchdog.restarts}
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <button
                onClick={() =>
                  window.dispatchEvent(new Event("openAuthDialog"))
                }
                className="text-xs text-primary hover:text-primary/80 transition-colors underline underline-offset-2 cursor-pointer"
              >
                Enter admin key
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Audio Library */}
      <div className="glass-panel p-6 rounded-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Music className="w-5 h-5 text-primary" />
          </div>
          <h3 className="text-lg font-display font-semibold text-white">
            Audio Library
          </h3>
          {audioStatus?.seeding_now && (
            <Badge
              variant="outline"
              className="ml-auto border-amber-500/50 text-amber-400 animate-pulse"
            >
              Seeding…
            </Badge>
          )}
          {!audioStatus?.seeding_now && audioStatus && (
            <>
              <Badge
                variant="outline"
                className={
                  (audioStatus.dataset?.num_chunks ?? 0) >=
                  (audioStatus.auto_growth?.threshold ?? 20)
                    ? "ml-auto border-green-500/40 text-green-400"
                    : "ml-auto border-amber-500/40 text-amber-400"
                }
              >
                {(audioStatus.dataset?.num_chunks ?? 0) >=
                (audioStatus.auto_growth?.threshold ?? 20)
                  ? "Healthy"
                  : "Growing"}
              </Badge>
              {(audioStatus.dataset?.num_chunks ?? 0) === 0 && (
                <Button
                  size="sm"
                  variant="outline"
                  className="border-primary/40 text-primary hover:bg-primary/10 ml-2"
                  disabled={seeding}
                  onClick={handleSeedAudio}
                >
                  {seeding ? "Seeding…" : "Seed Now"}
                </Button>
              )}
            </>
          )}
        </div>

        {audioStatus ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {/* Track count with progress bar */}
            <div className="col-span-2 sm:col-span-1 p-4 bg-black/20 rounded-xl border border-white/5 flex flex-col gap-2">
              <span className="text-xs text-muted-foreground">Tracks stored</span>
              <span className="text-3xl font-display font-bold text-white">
                {audioStatus.dataset?.num_chunks ?? 0}
              </span>
              {audioStatus.auto_growth?.threshold != null && (
                <>
                  <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        (audioStatus.dataset?.num_chunks ?? 0) >=
                        audioStatus.auto_growth.threshold
                          ? "bg-green-500"
                          : "bg-amber-500"
                      }`}
                      style={{
                        width: `${Math.min(
                          100,
                          ((audioStatus.dataset?.num_chunks ?? 0) /
                            audioStatus.auto_growth.threshold) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground">
                    threshold: {audioStatus.auto_growth.threshold}
                  </span>
                </>
              )}
            </div>

            {/* Last seeded */}
            <div className="p-4 bg-black/20 rounded-xl border border-white/5 flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Last seeded</span>
              <span className="text-sm font-mono text-white">
                {audioStatus.dataset?.seeded_at
                  ? new Date(
                      audioStatus.dataset.seeded_at * 1000
                    ).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "—"}
              </span>
            </div>

            {/* Auto-seed count */}
            <div className="p-4 bg-black/20 rounded-xl border border-white/5 flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Auto-seeds run</span>
              <span className="text-sm font-mono text-white">
                {audioStatus.auto_growth?.auto_seed_count ?? "—"}
              </span>
              {audioStatus.auto_growth?.last_auto_seed_at && (
                <span className="text-xs text-muted-foreground">
                  last:{" "}
                  {new Date(
                    audioStatus.auto_growth.last_auto_seed_at * 1000
                  ).toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              )}
            </div>

            {/* Source + interval */}
            <div className="p-4 bg-black/20 rounded-xl border border-white/5 flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Source</span>
              <span className="text-xs font-mono text-white truncate">
                {audioStatus.dataset?.source
                  ? audioStatus.dataset.source.replace("huggingface:", "HF — ").replace("librosa:", "librosa — ")
                  : "—"}
              </span>
              {audioStatus.auto_growth?.interval_hours != null && (
                <span className="text-xs text-muted-foreground mt-1">
                  checks every {audioStatus.auto_growth.interval_hours}h
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-[96px] bg-black/20 rounded-xl border border-white/5 border-dashed">
            <button
              onClick={() =>
                window.dispatchEvent(new Event("openAuthDialog"))
              }
              className="text-xs text-primary hover:text-primary/80 transition-colors underline underline-offset-2 cursor-pointer"
            >
              Enter admin key to view audio library status
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
