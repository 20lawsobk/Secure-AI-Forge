import { useGetGpuStatus, useGetHyperGpuStatus } from "@workspace/api-client-react";
import { Cpu, Server, Activity, ArrowUpRight } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Card } from "@/components/ui/card";

export default function GpuStatus() {
  const { data: gpu, isLoading: gpuLoading } = useGetGpuStatus();
  const { data: hyper, isLoading: hyperLoading } = useGetHyperGpuStatus();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold text-white">Compute Cluster</h1>
        <p className="text-muted-foreground mt-1">Live metrics from the digital and hyper GPU backends.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Digital GPU Card */}
        <Card className="glass-panel border-white/10 overflow-hidden">
          <div className="p-6 border-b border-white/5 bg-white/5 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Cpu className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h3 className="font-display font-bold text-white">Digital GPU</h3>
                <p className="text-xs text-muted-foreground">Standard Compute Backend</p>
              </div>
            </div>
            <div className={`px-3 py-1 rounded-full text-xs font-medium border ${gpu?.available ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
              {gpu?.available ? 'Online' : 'Offline'}
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            {gpuLoading ? (
              <Skeleton className="h-32 w-full bg-white/5" />
            ) : (
              <>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-muted-foreground">Core Utilization</span>
                    <span className="text-white font-mono">{gpu?.utilization || 0}%</span>
                  </div>
                  <Progress value={gpu?.utilization || 0} className="h-2 bg-white/5 [&>div]:bg-blue-500" />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black/30 p-4 rounded-xl">
                    <p className="text-xs text-muted-foreground">SIMD Lanes</p>
                    <p className="text-xl font-mono text-white mt-1">{gpu?.lanes || 0}</p>
                  </div>
                  <div className="bg-black/30 p-4 rounded-xl">
                    <p className="text-xs text-muted-foreground">VRAM Allocated</p>
                    <p className="text-xl font-mono text-white mt-1">{gpu?.vram_mb || 0} MB</p>
                  </div>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* Hyper GPU Card */}
        <Card className="glass-panel border-white/10 overflow-hidden relative">
          <div className="absolute top-0 right-0 p-32 bg-primary/10 blur-[100px] pointer-events-none rounded-full" />
          
          <div className="p-6 border-b border-white/5 bg-white/5 flex justify-between items-center relative z-10">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/20 rounded-lg">
                <Server className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="font-display font-bold text-white">HyperGPU Cluster</h3>
                <p className="text-xs text-muted-foreground">Accelerated Tensor Core Backend</p>
              </div>
            </div>
            <div className="px-3 py-1 rounded-full text-xs font-medium border bg-primary/10 text-primary border-primary/20 flex items-center gap-1">
              <Activity className="w-3 h-3" /> Active
            </div>
          </div>
          
          <div className="p-6 space-y-6 relative z-10">
            {hyperLoading ? (
              <Skeleton className="h-48 w-full bg-white/5" />
            ) : (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Throughput</p>
                    <p className="text-3xl font-display font-bold text-white mt-1">
                      {hyper?.total_tensor_core_tflops?.toFixed(2) || "0.00"} <span className="text-lg text-muted-foreground font-normal">TFLOPS</span>
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Compute Time</p>
                    <p className="text-3xl font-display font-bold text-white mt-1">
                      {hyper?.total_compute_ms ? (hyper.total_compute_ms / 1000).toFixed(1) : "0"} <span className="text-lg text-muted-foreground font-normal">s</span>
                    </p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                    <p className="text-[10px] uppercase text-muted-foreground">Lanes</p>
                    <p className="text-sm font-mono text-white mt-1">{hyper?.lanes}</p>
                  </div>
                  <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                    <p className="text-[10px] uppercase text-muted-foreground">Tensor Cores</p>
                    <p className="text-sm font-mono text-white mt-1">{hyper?.tensor_cores}</p>
                  </div>
                  <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                    <p className="text-[10px] uppercase text-muted-foreground">Precision</p>
                    <p className="text-sm font-mono text-primary-foreground mt-1">{hyper?.precision}</p>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-white/5 flex items-center justify-between text-xs text-muted-foreground">
                  <span>Engine: {hyper?.engine}</span>
                  <span className="flex items-center gap-1 text-primary-foreground cursor-pointer hover:text-primary transition-colors">
                    View Architecture <ArrowUpRight className="w-3 h-3" />
                  </span>
                </div>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
