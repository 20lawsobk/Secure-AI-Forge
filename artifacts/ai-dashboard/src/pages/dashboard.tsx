import { useQuery } from "@tanstack/react-query";
import { useGetDashboardStats, useHealthCheck } from "@workspace/api-client-react";
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
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

const BASE = "/api";

export default function Dashboard() {
  const { adminKey } = useAuth();
  const authHdr = adminKey ? { "X-Admin-Key": adminKey } : {};

  const { data: stats, isLoading: statsLoading } = useGetDashboardStats();
  const { data: health, isLoading: healthLoading } = useHealthCheck({
    query: { refetchInterval: 10000 },
  });

  const { data: storageStatus } = useQuery({
    queryKey: ["storage-status", adminKey],
    queryFn: async () => {
      const res = await fetch(`${BASE}/storage/status`, { headers: authHdr });
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    enabled: !!adminKey,
    refetchInterval: 15000,
    retry: false,
  });

  const { data: watchdog } = useQuery({
    queryKey: ["watchdog-status", adminKey],
    queryFn: async () => {
      const res = await fetch(`${BASE}/watchdog/status`, { headers: authHdr });
      if (!res.ok) throw new Error("failed");
      return res.json();
    },
    enabled: !!adminKey,
    refetchInterval: 20000,
    retry: false,
  });

  const uptimeFormatted = health?.uptime_seconds
    ? health.uptime_seconds >= 3600
      ? `${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m`
      : `${Math.floor(health.uptime_seconds / 60)}m ${health.uptime_seconds % 60}s`
    : null;

  const storageOnline = storageStatus?.connected === true;
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
          <h1 className="text-3xl font-display font-bold text-white">System Overview</h1>
          <p className="text-muted-foreground mt-1">Live metrics from the MaxCore AI cluster.</p>
        </div>

        <div className="glass-panel px-4 py-2 rounded-xl flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full ${health?.status === 'healthy' ? 'bg-green-500 animate-pulse' : 'bg-destructive'}`} />
            <span className="text-sm font-medium text-white capitalize">{health?.status || 'Unknown'}</span>
          </div>
          <div className="w-px h-4 bg-border" />
          <span className="text-xs text-muted-foreground font-mono">v{health?.version || '1.0.0'}</span>
          {uptimeFormatted && (
            <>
              <div className="w-px h-4 bg-border" />
              <span className="text-xs text-muted-foreground font-mono flex items-center gap-1">
                <Clock className="w-3 h-3" /> {uptimeFormatted}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Watchdog alerts banner */}
      {watchdogAlerts.length > 0 && (
        <div className="glass-panel p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <p className="text-sm font-medium text-amber-300">Watchdog Alerts</p>
            {watchdogAlerts.map((alert: string, i: number) => (
              <p key={i} className="text-xs text-amber-400/80">{alert}</p>
            ))}
          </div>
        </div>
      )}

      {/* Main Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Active API Keys"
          value={stats?.active_api_keys || 0}
          icon={KeySquare}
          description={`Out of ${stats?.total_api_keys || 0} total keys`}
        />
        <StatCard
          title="Requests Today"
          value={(stats?.total_requests_today || 0).toLocaleString()}
          icon={Activity}
          description="Across all endpoints"
        />
        <StatCard
          title="GPU Lanes"
          value={stats?.gpu_lanes || 0}
          icon={Cpu}
          description="HyperGPU Cluster"
        />
        <StatCard
          title="Vocab Size"
          value={(stats?.vocab_size || 0).toLocaleString()}
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
            <h3 className="text-lg font-display font-semibold text-white">Model Engine</h3>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
              <span className="text-sm text-muted-foreground">Engine Status</span>
              <Badge variant="outline" className={stats?.model_status === 'running' ? 'border-green-500/50 text-green-400' : ''}>
                {stats?.model_status || 'Unknown'}
              </Badge>
            </div>
            <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
              <span className="text-sm text-muted-foreground">Weights Loaded</span>
              <Badge variant="outline" className={stats?.weights_exist ? 'border-primary/50 text-primary-foreground' : 'border-destructive/50 text-destructive-foreground'}>
                {stats?.weights_exist ? 'Yes' : 'No'}
              </Badge>
            </div>
            <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
              <span className="text-sm text-muted-foreground">Training State</span>
              <div className="flex items-center gap-2">
                {stats?.training_state === 'running' && <Zap className="w-4 h-4 text-amber-400" />}
                <span className="text-sm font-medium text-white capitalize">{stats?.training_state || 'Idle'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Storage Status */}
        <div className="glass-panel p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-lg">
              <HardDrive className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">pdim Storage</h3>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
              <span className="text-sm text-muted-foreground">Connection</span>
              <Badge variant="outline" className={storageOnline ? 'border-green-500/50 text-green-400' : 'border-amber-500/50 text-amber-400'}>
                {storageOnline ? 'Online' : storageStatus ? 'Offline' : 'Checking…'}
              </Badge>
            </div>
            {storageStatus?.instance_id && (
              <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                <span className="text-sm text-muted-foreground">Instance</span>
                <span className="text-xs font-mono text-white truncate max-w-[120px]">{storageStatus.instance_id}</span>
              </div>
            )}
            {storageStatus?.keys_count != null && (
              <div className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                <span className="text-sm text-muted-foreground">Keys stored</span>
                <span className="text-sm font-mono text-white">{storageStatus.keys_count}</span>
              </div>
            )}
            {!storageStatus && (
              <div className="flex items-center justify-center h-[72px] bg-black/20 rounded-xl border border-white/5 border-dashed">
                <p className="text-xs text-muted-foreground">Requires admin key</p>
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
              <h3 className="text-lg font-display font-semibold text-white">Knowledge Base</h3>
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
              <h3 className="text-base font-display font-semibold text-white">Watchdog</h3>
              <Badge variant="outline" className={
                watchdog?.status === 'healthy' ? 'ml-auto border-green-500/40 text-green-400' :
                watchdogAlerts.length > 0 ? 'ml-auto border-amber-500/40 text-amber-400' :
                'ml-auto border-white/10 text-muted-foreground'
              }>
                {watchdog?.status || 'Unknown'}
              </Badge>
            </div>
            {watchdog ? (
              <div className="space-y-1.5 text-xs">
                {watchdog.checks_run != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Checks run</span>
                    <span className="text-white font-mono">{watchdog.checks_run}</span>
                  </div>
                )}
                {watchdog.restarts != null && (
                  <div className="flex justify-between text-muted-foreground">
                    <span>Auto-restarts</span>
                    <span className="text-white font-mono">{watchdog.restarts}</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Requires admin key</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
