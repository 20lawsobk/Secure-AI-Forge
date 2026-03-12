import { useGetDashboardStats, useHealthCheck } from "@workspace/api-client-react";
import { StatCard } from "@/components/stat-card";
import { 
  Server, 
  KeySquare, 
  Cpu, 
  Database, 
  Activity, 
  FileJson,
  Zap
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useGetDashboardStats();
  const { data: health, isLoading: healthLoading } = useHealthCheck({
    query: { refetchInterval: 10000 }
  });

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
        </div>
      </div>

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
          trend={{ value: 12.5, isPositive: true }}
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
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Server className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">Model Engine</h3>
          </div>
          
          <div className="space-y-4">
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

        <div className="glass-panel p-6 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-lg">
              <FileJson className="w-5 h-5 text-primary" />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">Knowledge Base</h3>
          </div>
          
          <div className="flex flex-col items-center justify-center h-[180px] bg-black/20 rounded-xl border border-white/5 border-dashed">
            <p className="text-4xl font-display font-bold text-white mb-2">
              {stats?.boostsheet_count || 0}
            </p>
            <p className="text-sm text-muted-foreground">Total BoostSheets</p>
            <p className="text-xs text-muted-foreground/50 mt-4 text-center px-6">
              BoostSheets act as the foundational training data for domain-specific models.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
