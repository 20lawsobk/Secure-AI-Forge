import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
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
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

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
        isRunning((query.state as any)?.data?.state) ? 3000 : 10000,
    },
  });

  const { data: logs, isLoading: logsLoading } = useGetTrainingLogs({
    query: { refetchInterval: () => (isRunning(status?.state) ? 3000 : 10000) },
  });

  const startMut = useMutation({
    mutationFn: () =>
      apiFetch("/training/start", {
        method: "POST",
        headers,
        body: JSON.stringify({
          epochs: 10,
          batch_size: 8,
          learning_rate: 0.0005,
        }),
      }),
    onSuccess: () => {
      toast({ title: "Training started successfully" });
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

  // ── Continuous Training ──
  const { data: contStatus, refetch: refetchCont } = useQuery({
    queryKey: ["continuous-status", adminKey],
    queryFn: () => apiFetch("/training/continuous/status", { headers }),
    enabled: !!adminKey,
    refetchInterval: 8000,
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

  // ── Data Puller ──
  const { data: pullerStatus, refetch: refetchPuller } = useQuery({
    queryKey: ["puller-status", adminKey],
    queryFn: () => apiFetch("/training/puller/status", { headers }),
    enabled: !!adminKey,
    refetchInterval: 10000,
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

  const handleStart = () => startMut.mutate();

  const progress =
    status?.total_epochs && status?.epoch
      ? (status.epoch / status.total_epochs) * 100
      : 0;

  const etaFormatted = status?.eta_seconds
    ? status.eta_seconds >= 3600
      ? `${Math.floor(status.eta_seconds / 3600)}h ${Math.floor((status.eta_seconds % 3600) / 60)}m`
      : `${Math.floor(status.eta_seconds / 60)}m ${status.eta_seconds % 60}s`
    : null;

  const contRunning = contStatus?.status === "running";
  const pullerRunning = pullerStatus?.status === "running";

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
        <div className="flex gap-3">
          <Button
            disabled={isRunning(status?.state) || startMut.isPending}
            onClick={handleStart}
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
                      {status?.samples_trained?.toLocaleString() || 0}
                    </span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">Time Elapsed</span>
                    <span className="text-white font-mono">
                      {status?.elapsed_seconds
                        ? `${Math.floor(status.elapsed_seconds / 60)}m ${status.elapsed_seconds % 60}s`
                        : "—"}
                    </span>
                  </div>
                  {etaFormatted && (
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
        </div>

        {/* Logs Terminal */}
        <div className="lg:col-span-2 glass-panel rounded-2xl flex flex-col overflow-hidden">
          <div className="bg-black/40 p-3 border-b border-white/5 flex items-center gap-2">
            <Terminal className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-mono text-muted-foreground">
              training_output.log
            </span>
          </div>
          <div className="p-4 bg-[#0a0a0c] flex-1 font-mono text-xs overflow-y-auto max-h-[600px] min-h-[400px]">
            {logsLoading ? (
              <div className="text-muted-foreground animate-pulse">
                Loading logs...
              </div>
            ) : logs?.logs.length === 0 ? (
              <div className="text-muted-foreground">
                No logs available for current session.
              </div>
            ) : (
              <div className="space-y-1.5">
                {logs?.logs.map((log, i) => (
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
