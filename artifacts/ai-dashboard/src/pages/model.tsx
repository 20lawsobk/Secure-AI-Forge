import { useGetModelStatus } from "@workspace/api-client-react";
import { BrainCircuit, Database, Layers, Hash, HardDrive } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function ModelStatus() {
  const { data: model, isLoading } = useGetModelStatus();

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-3xl font-display font-bold text-white">Model Architecture</h1>
        <p className="text-muted-foreground mt-1">Inspection of the core transformer structure and loaded weights.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="glass-panel border-white/10 p-6 flex flex-col justify-center min-h-[250px] relative overflow-hidden">
          <div className="absolute -right-10 -bottom-10 opacity-[0.05]">
            <BrainCircuit className="w-64 h-64" />
          </div>
          
          <h3 className="text-sm font-medium text-muted-foreground mb-6">Core Status</h3>
          
          {isLoading ? (
            <div className="space-y-4"><Skeleton className="h-10 w-full" /><Skeleton className="h-10 w-full" /></div>
          ) : (
            <div className="space-y-4 relative z-10">
              <div className="flex items-center justify-between p-4 bg-black/30 rounded-xl border border-white/5">
                <div className="flex items-center gap-3">
                  <Database className="w-5 h-5 text-primary" />
                  <span className="text-white font-medium">Weights Loaded</span>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-bold ${model?.weights_exist ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {model?.weights_exist ? 'YES' : 'NO'}
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-black/30 rounded-xl border border-white/5">
                <div className="flex items-center gap-3">
                  <HardDrive className="w-5 h-5 text-blue-400" />
                  <span className="text-white font-medium">Compute Device</span>
                </div>
                <div className="px-3 py-1 rounded-full text-xs font-bold uppercase bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  {model?.device || 'CPU'}
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card className="glass-panel border-white/10 p-6 min-h-[250px]">
          <h3 className="text-sm font-medium text-muted-foreground mb-6">Architecture Specs</h3>
          
          {isLoading ? (
            <div className="space-y-3"><Skeleton className="h-8 w-full" /><Skeleton className="h-8 w-full" /><Skeleton className="h-8 w-full" /></div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Hash className="w-4 h-4" /> <span className="text-xs uppercase">Vocab Size</span>
                </div>
                <p className="text-xl font-mono text-white">{model?.vocab_size?.toLocaleString()}</p>
              </div>
              
              <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Layers className="w-4 h-4" /> <span className="text-xs uppercase">Layers</span>
                </div>
                <p className="text-xl font-mono text-white">{model?.layers}</p>
              </div>

              <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <BrainCircuit className="w-4 h-4" /> <span className="text-xs uppercase">Heads</span>
                </div>
                <p className="text-xl font-mono text-white">{model?.heads}</p>
              </div>

              <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Database className="w-4 h-4" /> <span className="text-xs uppercase">Dimension</span>
                </div>
                <p className="text-xl font-mono text-white">{model?.dim}</p>
              </div>
              
              <div className="col-span-2 bg-black/20 p-3 rounded-lg border border-white/5 flex justify-between items-center">
                <span className="text-xs uppercase text-muted-foreground">Max Sequence Length</span>
                <span className="text-sm font-mono text-white">{model?.max_len} tokens</span>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
