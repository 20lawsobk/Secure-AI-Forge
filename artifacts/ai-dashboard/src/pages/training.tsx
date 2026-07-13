import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  useGetTrainingStatus,
  useGetTrainingLogs,
} from "@workspace/api-client-react";
import { format } from "date-fns";
import {
  Play,
  Square,
  Terminal,
  Zap,
  RefreshCw,
  Database,
  Clock,
  RotateCcw,
  ChevronDown,
  Music,
  TrendingUp,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatUptime } from "@/lib/format";

const BASE = "/api";

function authHeaders(adminKey: string | null) {
  return adminKey ? { "X-Admin-Key": adminKey } : {};
}

async function apiFetch(path: string, opts: RequestInit = {}) {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export default function Training() {
  const { toast } = useToast();
  const { adminKey } = useAuth();
  const qc = useQueryClient();
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.0005);

  const headers = {
    "Content-Type": "application/json",
    ...authHeaders(adminKey),
  } as HeadersInit;

  const isRunning = (state: string | undefined) => state === "running";

  const {
    data: status,
    isLoading: statusLoading,
    refetch: refetchStatus,
  } = useGetTrainingStatus({
    query: {
      refetchInterval: (query) =>
        isRunning((query.state as any)?.data?.state) ? 3000 : 10_000,
    },
  });

  const { data: logs, isLoading: logsLoading } = useGetTrainingLogs({
    query: {
      refetchInterval: () => (isRunning(status?.state) ? 3000 : 15_000),
    },
  });

  useEffect(() => {
    if (!autoScroll || !logContainerRef.current) return;
    const el = logContainerRef.current;
    el.scrollTop = el.scrollHeight;
  }, [logs, autoScroll]);

  const handleScroll = () => {
    const el = logContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setAutoScroll(atBottom);
  };

  const startMut = useMutation({
    mutationFn: () =>
      apiFetch("/training/start", {
        method: "POST",
        headers,
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
        }),
      }),
    onSuccess: () => {
      toast({ title: "Training started" });
      refetchStatus();
    },
    onError: () =>
      toast({ variant: "destructive", title: "Failed to start training" }),
  });

  const stopMut = useMutation({
    mutationFn: () => apiFetch("/training/stop", { method: "POST", headers }),
    onSuccess: () => {
      toast({ title: "Training stopped" });
      refetchStatus();
    },
    onError: () =>
      toast({ variant: "destructive", title: "Failed to stop training" }),
  });

  const { data: contStatus, refetch: refetchCont } = useQuery({
    queryKey: ["continuous-status", adminKey],
    queryFn: () => apiFetch("/training/continuous/status", { headers }),
    enabled: !!adminKey,
    refetchInterval: 8_000,
  });

  const contStartMut = useMutation({
    mutationFn: () =>
      apiFetch("/training/continuous/start", { method: "POST", headers }),
    onSuccess: () => {
      toast({ title: "Continuous training started" });
      refetchCont();
    },
    onError: () =>
      toast({
        variant: "destructive",
        title: "Failed to start continuous training",
      }),
  });

  const contStopMut = useMutation({
    mutationFn: () =>
      apiFetch("/training/continuous/stop", { method: "POST", headers }),
    onSuccess: () => {
      toast({ title: "Continuous training stopped" });
      refetchCont();
    },
    onError: () =>
      toast({
        variant: "destructive",
        title: "Failed to stop continuous training",
      }),
  });

  const { data: pullerStatus, refetch: refetchPuller } = useQuery({
    queryKey: ["puller-status", adminKey],
    queryFn: () => apiFetch("/training/puller/status", { headers }),
    enabled: !!adminKey,
    refetchInterval: 12_000,
  });

  const pullNowMut = useMutation({
    mutationFn: () =>
      apiFetch("/training/puller/pull", { method: "POST", headers }),
    onSuccess: () => {
      toast({ title: "Data pull triggered" });
      refetchPuller();
    },
    onError: () => toast({ variant: "destructive", title: "Pull failed" }),
  });

  // Audio dataset state
  const [audioSource, setAudioSource] = useState<"auto" | "hf" | "librosa">("auto");

  const { data: audioStatus, refetch: refetchAudio } = useQuery({
    queryKey: ["audio-dataset-status", adminKey],
    queryFn: () => apiFetch("/storage/datasets/audio/status", { headers }),
    enabled: !!adminKey,
    refetchInterval: 20_000,
  });

  const topUpMut = useMutation({
    mutationFn: () =>
      apiFetch(
        `/storage/datasets/audio/seed?count=6&replace=false${audioSource !== "auto" ? `&source=${audioSource}` : ""}`,
        { method: "POST", headers }
      ),
    onSuccess: () => {
      toast({ title: "Audio top-up started", description: "6 tracks are being added in the background." });
      setTimeout(() => refetchAudio(), 3000);
    },
    onError: () =>
      toast({ variant: "destructive", title: "Top-up failed", description: "Storage may be unavailable." }),
  });

  const progress =
    status?.total_epochs && status?.epoch
      ? (status.epoch / status.total_epochs) * 100
      : 0;

  const etaFormatted = formatUptime(status?.eta_seconds);
  const elapsedFormatted = formatUptime(status?.elapsed_seconds);

  const contRunning = contStatus?.status === "running";
  const pullerRunning = pullerStatus?.status === "running";
  const logList = logs?.logs ?? [];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-display font-bold text-white">
            Model Training
          </h1>
          <p className="text-muted-foreground mt-1">
            Monitor and control transformer curriculum training.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <label className="flex items-center gap-1">
              Epochs
              <input
                type="number"
                min={1}
                max={100}
                value={epochs}
                disabled={isRunning(status?.state)}
                onChange={(e) =>
                  setEpochs(Math.max(1, parseInt(e.target.value) || 1))
                }
                className="w-14 ml-1 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-white text-xs text-center disabled:opacity-40 focus:outline-none focus:border-primary"
              />
            </label>
            <label className="flex items-center gap-1">
              Batch
              <input
                type="number"
                min={1}
                max={256}
                value={batchSize}
                disabled={isRunning(status?.state)}
                onChange={(e) =>
                  setBatchSize(Math.max(1, parseInt(e.target.value) || 1))
                }
                className="w-14 ml-1 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-white text-xs text-center disabled:opacity-40 focus:outline-none focus:border-primary"
              />
            </label>
            <label className="flex items-center gap-1">
              LR
              <input
                type="number"
                min={0.00001}
                max={0.1}
                step={0.0001}
                value={learningRate}
                disabled={isRunning(status?.state)}
                onChange={(e) =>
                  setLearningRate(parseFloat(e.target.value) || 0.0005)
                }
                className="w-20 ml-1 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-white text-xs text-center disabled:opacity-40 focus:outline-none focus:border-primary"
              />
            </label>
          </div>
          <div className="flex gap-3">
            <Button
              disabled={isRunning(status?.state) || startMut.isPending}
              onClick={() => startMut.mutate()}
              className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20"
            >
              {startMut.isPending ? (
                "Starting..."
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" /> Start Run
                </>
              )}
            </Button>
            <Button
              variant="destructive"
              disabled={!isRunning(status?.state) || stopMut.isPending}
              onClick={() => stopMut.mutate()}
              className="shadow-lg shadow-destructive/20"
            >
              {stopMut.isPending ? (
                "Stopping..."
              ) : (
                <>
                  <Square className="w-4 h-4 mr-2" /> Stop
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Status Panel */}
        <div className="lg:col-span-1 space-y-4">
          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-display font-semibold text-white mb-4 flex items-center gap-2">
              <Zap
                className={`w-5 h-5 ${isRunning(status?.state) ? "text-amber-400 animate-pulse" : "text-muted-foreground"}`}
              />
              Current Session
            </h3>

            {statusLoading ? (
              <div className="space-y-4">
                <Skeleton className="h-8 w-full bg-white/5" />
                <Skeleton className="h-8 w-full bg-white/5" />
              </div>
            ) : (
              <div className="space-y-5">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="text-white font-medium">
                      Epoch {status?.epoch || 0} / {status?.total_epochs || 0}
                    </span>
                  </div>
                  <Progress value={progress} className="h-2 bg-white/10" />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                    <p className="text-xs text-muted-foreground mb-1">Loss</p>
                    <p className="text-2xl font-mono text-white">
                      {status?.loss ? status.loss.toFixed(4) : "—"}
                    </p>
                  </div>
                  <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                    <p className="text-xs text-muted-foreground mb-1">
                      Perplexity
                    </p>
                    <p className="text-2xl font-mono text-primary-foreground">
                      {status?.perplexity ? status.perplexity.toFixed(2) : "—"}
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">State</span>
                    <Badge
                      variant={
                        isRunning(status?.state) ? "default" : "secondary"
                      }
                      className={
                        isRunning(status?.state)
                          ? "bg-amber-500/20 text-amber-400 hover:bg-amber-500/20"
                          : ""
                      }
                    >
                      {status?.state || "Idle"}
                    </Badge>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">
                      Samples Trained
                    </span>
                    <span className="text-white font-mono">
                      {(status?.samples_trained ?? 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">Time Elapsed</span>
                    <span className="text-white font-mono">
                      {elapsedFormatted}
                    </span>
                  </div>
                  {status?.eta_seconds != null && (
                    <div className="flex justify-between p-2 rounded bg-primary/10 border border-primary/20 text-sm">
                      <span className="text-primary-foreground flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5" /> ETA
                      </span>
                      <span className="text-white font-mono">
                        {etaFormatted}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Continuous Training */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-display font-semibold text-white flex items-center gap-2">
                <RotateCcw
                  className={`w-4 h-4 ${contRunning ? "text-green-400 animate-spin" : "text-muted-foreground"}`}
                />
                Continuous Training
              </h3>
              <Badge
                variant="outline"
                className={
                  contRunning
                    ? "border-green-500/40 text-green-400"
                    : "border-white/10 text-muted-foreground"
                }
              >
                {contRunning ? "Running" : contStatus?.status || "Idle"}
              </Badge>
            </div>
            {contStatus && (
              <div className="space-y-1.5 text-xs">
                {contStatus.cycle_count != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Cycles completed</span>
                    <span className="text-white font-mono">
                      {contStatus.cycle_count}
                    </span>
                  </div>
                )}
                {contStatus.last_loss != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Last loss</span>
                    <span className="text-white font-mono">
                      {Number(contStatus.last_loss).toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
            )}
            <div className="flex gap-2 pt-1">
              <Button
                size="sm"
                disabled={contRunning || contStartMut.isPending}
                onClick={() => contStartMut.mutate()}
                className="flex-1 h-8 text-xs bg-green-600/20 hover:bg-green-600/30 text-green-300 border border-green-600/30"
              >
                <Play className="w-3 h-3 mr-1" /> Start
              </Button>
              <Button
                size="sm"
                disabled={!contRunning || contStopMut.isPending}
                onClick={() => contStopMut.mutate()}
                className="flex-1 h-8 text-xs bg-destructive/20 hover:bg-destructive/30 text-destructive border border-destructive/30"
              >
                <Square className="w-3 h-3 mr-1" /> Stop
              </Button>
            </div>
          </div>

          {/* Data Puller */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-display font-semibold text-white flex items-center gap-2">
                <Database
                  className={`w-4 h-4 ${pullerRunning ? "text-blue-400 animate-pulse" : "text-muted-foreground"}`}
                />
                Data Puller
              </h3>
              <Badge
                variant="outline"
                className={
                  pullerRunning
                    ? "border-blue-500/40 text-blue-400"
                    : "border-white/10 text-muted-foreground"
                }
              >
                {pullerRunning ? "Pulling" : pullerStatus?.status || "Idle"}
              </Badge>
            </div>
            {pullerStatus && (
              <div className="space-y-1.5 text-xs">
                {pullerStatus.samples_pulled != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Samples pulled</span>
                    <span className="text-white font-mono">
                      {Number(pullerStatus.samples_pulled).toLocaleString()}
                    </span>
                  </div>
                )}
                {pullerStatus.last_pull_at && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Last pull</span>
                    <span className="text-white font-mono">
                      {format(new Date(pullerStatus.last_pull_at), "HH:mm:ss")}
                    </span>
                  </div>
                )}
              </div>
            )}
            <Button
              size="sm"
              disabled={pullNowMut.isPending}
              onClick={() => pullNowMut.mutate()}
              className="w-full h-8 text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-600/30"
            >
              <RefreshCw
                className={`w-3 h-3 mr-1 ${pullNowMut.isPending ? "animate-spin" : ""}`}
              />
              {pullNowMut.isPending ? "Pulling..." : "Pull Now"}
            </Button>
          </div>

          {/* Audio Dataset */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-display font-semibold text-white flex items-center gap-2">
                <Music
                  className={`w-4 h-4 ${audioStatus?.seeding_now ? "text-purple-400 animate-pulse" : "text-muted-foreground"}`}
                />
                Audio Dataset
              </h3>
              <Badge
                variant="outline"
                className={
                  audioStatus?.seeding_now
                    ? "border-purple-500/40 text-purple-400"
                    : (audioStatus?.dataset?.num_chunks ?? 0) >= 20
                    ? "border-green-500/40 text-green-400"
                    : "border-amber-500/40 text-amber-400"
                }
              >
                {audioStatus?.seeding_now
                  ? "Seeding…"
                  : `${audioStatus?.dataset?.num_chunks ?? 0} tracks`}
              </Badge>
            </div>

            {audioStatus && (
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between text-muted-foreground">
                  <span>Total tracks</span>
                  <span className="text-white font-mono">
                    {audioStatus.dataset?.num_chunks ?? 0}
                    <span className="text-muted-foreground ml-1">/ {audioStatus.auto_growth?.threshold ?? 20} threshold</span>
                  </span>
                </div>
                {audioStatus.dataset?.source && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Source</span>
                    <span className="text-white font-mono truncate max-w-[140px]" title={audioStatus.dataset.source}>
                      {audioStatus.dataset.source.startsWith("huggingface") ? "HuggingFace FMA" : "librosa examples"}
                    </span>
                  </div>
                )}
                {audioStatus.dataset?.seeded_at && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Last seeded</span>
                    <span className="text-white font-mono">
                      {format(new Date(audioStatus.dataset.seeded_at * 1000), "MM/dd HH:mm")}
                    </span>
                  </div>
                )}
                {audioStatus.auto_growth?.enabled && (
                  <div className="flex justify-between text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" /> Auto-grows
                    </span>
                    <span className="text-white font-mono">
                      {audioStatus.auto_growth.auto_seed_count ?? 0}× so far
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Source selector + Top-Up button */}
            <div className="flex gap-2 pt-1">
              <select
                value={audioSource}
                onChange={(e) => setAudioSource(e.target.value as typeof audioSource)}
                className="flex-1 h-8 text-xs rounded bg-white/5 border border-white/10 text-white px-2 focus:outline-none focus:border-purple-500/50"
              >
                <option value="auto">Auto source</option>
                <option value="hf">Force HuggingFace</option>
                <option value="librosa">Force librosa</option>
              </select>
              <Button
                size="sm"
                disabled={topUpMut.isPending || audioStatus?.seeding_now}
                onClick={() => topUpMut.mutate()}
                className="h-8 text-xs bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 border border-purple-600/30 px-3"
              >
                <RefreshCw
                  className={`w-3 h-3 mr-1 ${topUpMut.isPending ? "animate-spin" : ""}`}
                />
                {topUpMut.isPending || audioStatus?.seeding_now ? "Seeding…" : "Top Up"}
              </Button>
            </div>
          </div>
        </div>

        {/* Logs Terminal */}
        <div className="lg:col-span-2 glass-panel rounded-2xl flex flex-col overflow-hidden">
          <div className="bg-black/40 p-3 border-b border-white/5 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm font-mono text-muted-foreground">
                training_output.log
              </span>
              {logList.length > 0 && (
                <span className="text-xs text-muted-foreground/50 font-mono">
                  ({logList.length} entries)
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {!autoScroll && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs text-muted-foreground hover:text-white px-2"
                  onClick={() => {
                    setAutoScroll(true);
                    logContainerRef.current?.scrollTo({
                      top: logContainerRef.current.scrollHeight,
                      behavior: "smooth",
                    });
                  }}
                >
                  <ChevronDown className="w-3 h-3 mr-1" />
                  Jump to bottom
                </Button>
              )}
            </div>
          </div>
          <div
            ref={logContainerRef}
            onScroll={handleScroll}
            className="p-4 bg-[#0a0a0c] flex-1 font-mono text-xs overflow-y-auto max-h-[600px] min-h-[400px] scroll-smooth"
          >
            {logsLoading ? (
              <div className="text-muted-foreground animate-pulse">
                Loading logs...
              </div>
            ) : logList.length === 0 ? (
              <div className="text-muted-foreground">
                No logs available for current session.
              </div>
            ) : (
              <div className="space-y-1.5">
                {logList.map((log, i) => (
                  <div
                    key={i}
                    className="flex gap-3 hover:bg-white/5 p-1 rounded transition-colors"
                  >
                    <span className="text-muted-foreground/50 shrink-0 w-20">
                      {format(new Date(log.timestamp), "HH:mm:ss")}
                    </span>
                    <span
                      className={`shrink-0 w-12 ${
                        log.level === "error"
                          ? "text-destructive"
                          : log.level === "warn"
                            ? "text-amber-400"
                            : "text-blue-400"
                      }`}
                    >
                      [{log.level.toUpperCase()}]
                    </span>
                    <span className="text-gray-300 break-all">
                      {log.message}
                    </span>
                    {log.loss && (
                      <span className="ml-auto text-primary-foreground shrink-0 pl-4">
                        loss={log.loss.toFixed(4)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
