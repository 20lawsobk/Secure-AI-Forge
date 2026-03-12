import { useState } from "react";
import { format } from "date-fns";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Copy, Plus, Trash2, RefreshCw, Key, ShieldAlert } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import {
  useListApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
  useRotateApiKey,
} from "@workspace/api-client-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Skeleton } from "@/components/ui/skeleton";

const SCOPES = ["read", "write", "train", "admin", "generate"];

const createKeySchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  scopes: z.array(z.string()).min(1, "Select at least one scope"),
  expires_in_days: z.coerce.number().optional().nullable(),
});

export default function ApiKeys() {
  const { adminKey } = useAuth();
  const { toast } = useToast();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [newKeyData, setNewKeyData] = useState<{ key: string; name: string } | null>(null);

  const { data, isLoading, refetch } = useListApiKeys({
    query: { enabled: !!adminKey },
  });

  const createMut = useCreateApiKey();
  const revokeMut = useRevokeApiKey();
  const rotateMut = useRotateApiKey();

  const form = useForm<z.infer<typeof createKeySchema>>({
    resolver: zodResolver(createKeySchema),
    defaultValues: {
      name: "",
      scopes: ["read", "generate"],
      expires_in_days: null,
    },
  });

  const onSubmit = async (values: z.infer<typeof createKeySchema>) => {
    try {
      const result = await createMut.mutateAsync({ data: values });
      setNewKeyData({ key: result.key, name: result.name });
      setIsCreateOpen(false);
      form.reset();
      refetch();
      toast({ title: "API Key created successfully" });
    } catch (e) {
      toast({ variant: "destructive", title: "Failed to create key" });
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm("Are you sure you want to revoke this key?")) return;
    try {
      await revokeMut.mutateAsync({ keyId: id });
      refetch();
      toast({ title: "Key revoked" });
    } catch (e) {
      toast({ variant: "destructive", title: "Failed to revoke key" });
    }
  };

  const handleRotate = async (id: string) => {
    if (!confirm("Rotating will invalidate the old key immediately. Continue?")) return;
    try {
      const result = await rotateMut.mutateAsync({ keyId: id });
      setNewKeyData({ key: result.key, name: result.name });
      refetch();
      toast({ title: "Key rotated" });
    } catch (e) {
      toast({ variant: "destructive", title: "Failed to rotate key" });
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: "Copied to clipboard" });
  };

  if (!adminKey) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <div className="w-16 h-16 rounded-2xl bg-destructive/10 flex items-center justify-center mb-4">
          <ShieldAlert className="w-8 h-8 text-destructive" />
        </div>
        <h2 className="text-2xl font-display font-bold text-white mb-2">Authentication Required</h2>
        <p className="text-muted-foreground max-w-md">
          You need to provide an Admin Key in the sidebar to view and manage API keys.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-display font-bold text-white">API Keys</h1>
          <p className="text-muted-foreground mt-1">Manage access tokens for external integrations.</p>
        </div>
        <Button 
          onClick={() => setIsCreateOpen(true)}
          className="bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20"
        >
          <Plus className="w-4 h-4 mr-2" /> Create New Key
        </Button>
      </div>

      <div className="glass-panel rounded-2xl overflow-hidden">
        <Table>
          <TableHeader className="bg-black/20">
            <TableRow className="border-white/5 hover:bg-transparent">
              <TableHead className="text-white">Name</TableHead>
              <TableHead className="text-white">Prefix</TableHead>
              <TableHead className="text-white">Scopes</TableHead>
              <TableHead className="text-white">Created</TableHead>
              <TableHead className="text-white">Last Used</TableHead>
              <TableHead className="text-white text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              [...Array(3)].map((_, i) => (
                <TableRow key={i} className="border-white/5">
                  <TableCell><Skeleton className="h-5 w-32 bg-white/5" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-24 bg-white/5" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-40 bg-white/5" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-24 bg-white/5" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-24 bg-white/5" /></TableCell>
                  <TableCell><Skeleton className="h-8 w-20 bg-white/5 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : data?.keys.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="h-32 text-center text-muted-foreground">
                  No API keys found. Create one to get started.
                </TableCell>
              </TableRow>
            ) : (
              data?.keys.map((key) => (
                <TableRow key={key.id} className="border-white/5 hover:bg-white/5">
                  <TableCell className="font-medium text-white">{key.name}</TableCell>
                  <TableCell className="font-mono text-muted-foreground">{key.prefix}••••••••</TableCell>
                  <TableCell>
                    <div className="flex gap-1 flex-wrap">
                      {key.scopes.map(s => (
                        <Badge key={s} variant="secondary" className="bg-white/5 text-xs font-normal border-white/10">
                          {s}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {format(new Date(key.created_at), "MMM d, yyyy")}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {key.last_used_at ? format(new Date(key.last_used_at), "MMM d, HH:mm") : "Never"}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-white" onClick={() => handleRotate(key.id)} title="Rotate Key">
                        <RefreshCw className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive hover:text-red-400 hover:bg-destructive/10" onClick={() => handleRevoke(key.id)} title="Revoke Key">
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Create Key Dialog */}
      <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
        <DialogContent className="sm:max-w-md glass-panel border-white/10">
          <DialogHeader>
            <DialogTitle className="text-xl font-display text-white">Create API Key</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Generate a new key for API access. The key will only be shown once.
            </DialogDescription>
          </DialogHeader>

          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 py-4">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-white">Key Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g. Production Backend" className="bg-black/50 border-white/10 text-white" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="scopes"
                render={() => (
                  <FormItem>
                    <div className="mb-4">
                      <FormLabel className="text-white">Scopes</FormLabel>
                      <FormDescription className="text-xs">Select permissions for this key.</FormDescription>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {SCOPES.map((scope) => (
                        <FormField
                          key={scope}
                          control={form.control}
                          name="scopes"
                          render={({ field }) => {
                            return (
                              <FormItem
                                key={scope}
                                className="flex flex-row items-start space-x-3 space-y-0 rounded-md border border-white/10 bg-black/20 p-3"
                              >
                                <FormControl>
                                  <Checkbox
                                    checked={field.value?.includes(scope)}
                                    onCheckedChange={(checked) => {
                                      return checked
                                        ? field.onChange([...field.value, scope])
                                        : field.onChange(
                                            field.value?.filter((value) => value !== scope)
                                          )
                                    }}
                                  />
                                </FormControl>
                                <FormLabel className="font-normal text-sm capitalize text-white cursor-pointer">
                                  {scope}
                                </FormLabel>
                              </FormItem>
                            )
                          }}
                        />
                      ))}
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <DialogFooter className="pt-4">
                <Button variant="ghost" type="button" onClick={() => setIsCreateOpen(false)}>Cancel</Button>
                <Button type="submit" disabled={createMut.isPending} className="bg-primary text-white">
                  {createMut.isPending ? "Creating..." : "Create Key"}
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Show New Key Dialog */}
      <Dialog open={!!newKeyData} onOpenChange={() => setNewKeyData(null)}>
        <DialogContent className="sm:max-w-md glass-panel border-white/10">
          <DialogHeader>
            <DialogTitle className="text-xl font-display text-white flex items-center gap-2">
              <Key className="w-5 h-5 text-primary" /> Key Generated
            </DialogTitle>
            <DialogDescription className="text-amber-400 font-medium">
              Please copy this key now. You won't be able to see it again!
            </DialogDescription>
          </DialogHeader>
          
          <div className="p-4 bg-black/50 rounded-xl border border-white/10 mt-2 relative group">
            <code className="text-primary-foreground font-mono text-sm break-all pr-10 block">
              {newKeyData?.key}
            </code>
            <Button
              size="icon"
              variant="ghost"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-white hover:bg-white/10"
              onClick={() => copyToClipboard(newKeyData?.key || "")}
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          
          <DialogFooter className="mt-4">
            <Button onClick={() => setNewKeyData(null)} className="w-full">
              I've copied it safely
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
