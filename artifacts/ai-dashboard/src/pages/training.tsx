import { useGetTrainingStatus, useGetTrainingLogs, useStartTraining } from "@workspace/api-client-react";
import { format } from "date-fns";
import { Play, Square, Terminal, Zap, CheckCircle2, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export default function Training() {
  const { toast } = useToast();
  
  // Polling more aggressively if state is running
  const { data: status, isLoading: statusLoading, refetch: refetchStatus } = useGetTrainingStatus({
    query: { refetchInterval: (data) => data?.state === 'running' ? 3000 : 10000 }
  });

  const { data: logs, isLoading: logsLoading } = useGetTrainingLogs({
    query: { refetchInterval: (data) => status?.state === 'running' ? 3000 : 10000 }
  });

  const startMut = useStartTraining();

  const isRunning = status?.state === "running";

  const handleStart = async () => {
    try {
      await startMut.mutateAsync({ data: { epochs: 10, batch_size: 8, learning_rate: 0.0005 } });
      toast({ title: "Training started successfully" });
      refetchStatus();
    } catch (e) {
      toast({ variant: "destructive", title: "Failed to start training" });
    }
  };

  const progress = status?.total_epochs && status?.epoch 
    ? (status.epoch / status.total_epochs) * 100 
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-display font-bold text-white">Model Training</h1>
          <p className="text-muted-foreground mt-1">Monitor and control transformer curriculum training.</p>
        </div>
        <div className="flex gap-3">
          <Button 
            disabled={isRunning || startMut.isPending} 
            onClick={handleStart}
            className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20"
          >
            {startMut.isPending ? "Starting..." : <><Play className="w-4 h-4 mr-2" /> Start Run</>}
          </Button>
          <Button 
            variant="destructive" 
            disabled={!isRunning}
            className="shadow-lg shadow-destructive/20"
          >
            <Square className="w-4 h-4 mr-2" /> Stop
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Status Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-display font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className={`w-5 h-5 ${isRunning ? 'text-amber-400 animate-pulse' : 'text-muted-foreground'}`} />
              Current Session
            </h3>
            
            {statusLoading ? (
              <div className="space-y-4"><Skeleton className="h-8 w-full bg-white/5" /><Skeleton className="h-8 w-full bg-white/5" /></div>
            ) : (
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="text-white font-medium">Epoch {status?.epoch || 0} / {status?.total_epochs || 0}</span>
                  </div>
                  <Progress value={progress} className="h-2 bg-white/10" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                    <p className="text-xs text-muted-foreground mb-1">Loss</p>
                    <p className="text-2xl font-mono text-white">{status?.loss ? status.loss.toFixed(4) : "—"}</p>
                  </div>
                  <div className="bg-black/20 p-4 rounded-xl border border-white/5">
                    <p className="text-xs text-muted-foreground mb-1">Perplexity</p>
                    <p className="text-2xl font-mono text-primary-foreground">{status?.perplexity ? status.perplexity.toFixed(2) : "—"}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">State</span>
                    <Badge variant={isRunning ? "default" : "secondary"} className={isRunning ? "bg-amber-500/20 text-amber-400 hover:bg-amber-500/20" : ""}>
                      {status?.state || "Idle"}
                    </Badge>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">Samples Trained</span>
                    <span className="text-white font-mono">{status?.samples_trained?.toLocaleString() || 0}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-white/5 text-sm">
                    <span className="text-muted-foreground">Time Elapsed</span>
                    <span className="text-white font-mono">{status?.elapsed_seconds ? `${Math.floor(status.elapsed_seconds / 60)}m ${status.elapsed_seconds % 60}s` : "—"}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Logs Terminal */}
        <div className="lg:col-span-2 glass-panel rounded-2xl flex flex-col overflow-hidden">
          <div className="bg-black/40 p-3 border-b border-white/5 flex items-center gap-2">
            <Terminal className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-mono text-muted-foreground">training_output.log</span>
          </div>
          <div className="p-4 bg-[#0a0a0c] flex-1 font-mono text-xs overflow-y-auto max-h-[600px] min-h-[400px]">
            {logsLoading ? (
              <div className="text-muted-foreground animate-pulse">Loading logs...</div>
            ) : logs?.logs.length === 0 ? (
              <div className="text-muted-foreground">No logs available for current session.</div>
            ) : (
              <div className="space-y-1.5">
                {logs?.logs.map((log, i) => (
                  <div key={i} className="flex gap-3 hover:bg-white/5 p-1 rounded transition-colors">
                    <span className="text-muted-foreground/50 shrink-0 w-20">
                      {format(new Date(log.timestamp), "HH:mm:ss")}
                    </span>
                    <span className={`shrink-0 w-12 ${
                      log.level === 'error' ? 'text-destructive' : 
                      log.level === 'warn' ? 'text-amber-400' : 'text-blue-400'
                    }`}>
                      [{log.level.toUpperCase()}]
                    </span>
                    <span className="text-gray-300 break-all">{log.message}</span>
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
