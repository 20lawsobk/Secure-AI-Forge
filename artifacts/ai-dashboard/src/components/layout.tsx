import { Link, useLocation } from "wouter";
import { 
  Activity, 
  Key, 
  Cpu, 
  BrainCircuit, 
  PenTool, 
  Settings,
  Menu,
  Lock,
  LogOut,
  ChevronRight
} from "lucide-react";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogDescription,
  DialogFooter
} from "@/components/ui/dialog";

const navItems = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/api-keys", label: "API Keys", icon: Key },
  { href: "/training", label: "Training", icon: BrainCircuit },
  { href: "/gpu", label: "GPU Cluster", icon: Cpu },
  { href: "/content", label: "Generators", icon: PenTool },
  { href: "/model", label: "Model Status", icon: Settings },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const { adminKey, setAdminKey, clearAdminKey } = useAuth();
  const [isAuthDialogOpen, setIsAuthDialogOpen] = useState(!adminKey);
  const [keyInput, setKeyInput] = useState("");

  const handleSaveKey = () => {
    setAdminKey(keyInput);
    setIsAuthDialogOpen(false);
  };

  return (
    <div className="min-h-screen bg-background flex overflow-hidden selection:bg-primary/30">
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        {isSidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="flex-shrink-0 border-r border-border/50 bg-sidebar flex flex-col relative z-20"
          >
            <div className="p-6 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-fuchsia-600 flex items-center justify-center shadow-lg shadow-primary/20">
                <BrainCircuit className="w-5 h-5 text-white" />
              </div>
              <span className="font-display font-bold text-xl tracking-tight text-white">MaxCore AI</span>
            </div>

            <nav className="flex-1 px-4 py-4 space-y-1">
              <div className="text-xs font-semibold text-sidebar-foreground/50 uppercase tracking-wider mb-4 px-2">
                Platform
              </div>
              {navItems.map((item) => {
                const isActive = location === item.href;
                const Icon = item.icon;
                return (
                  <Link key={item.href} href={item.href}>
                    <div
                      className={`
                        flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-200 group
                        ${isActive 
                          ? "bg-primary/10 text-primary font-medium" 
                          : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-white"
                        }
                      `}
                    >
                      <Icon className={`w-5 h-5 ${isActive ? "text-primary" : "text-sidebar-foreground group-hover:text-white transition-colors"}`} />
                      <span>{item.label}</span>
                      {isActive && (
                        <motion.div layoutId="activeNav" className="ml-auto">
                          <ChevronRight className="w-4 h-4 text-primary" />
                        </motion.div>
                      )}
                    </div>
                  </Link>
                );
              })}
            </nav>

            <div className="p-4 border-t border-border/50">
              <div className="glass-panel p-4 rounded-xl">
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-2 h-2 rounded-full ${adminKey ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.6)]' : 'bg-destructive shadow-[0_0_10px_rgba(239,68,68,0.6)]'}`} />
                  <span className="text-sm font-medium text-white">
                    {adminKey ? "Authenticated" : "Unauthenticated"}
                  </span>
                </div>
                <Button 
                  variant="secondary" 
                  size="sm" 
                  className="w-full text-xs mt-2 bg-white/5 hover:bg-white/10 text-white border-white/10"
                  onClick={() => adminKey ? clearAdminKey() : setIsAuthDialogOpen(true)}
                >
                  {adminKey ? (
                    <><LogOut className="w-3 h-3 mr-2" /> Remove Key</>
                  ) : (
                    <><Lock className="w-3 h-3 mr-2" /> Set Admin Key</>
                  )}
                </Button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden relative">
        {/* Top Header */}
        <header className="h-16 flex items-center justify-between px-6 border-b border-border/50 bg-background/80 backdrop-blur-md z-10 sticky top-0">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="text-muted-foreground hover:text-white">
              <Menu className="w-5 h-5" />
            </Button>
            <div className="h-4 w-px bg-border"></div>
            <h2 className="text-sm font-medium text-muted-foreground capitalize">
              {location === "/" ? "Overview" : location.replace("/", "").replace("-", " ")}
            </h2>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 text-xs font-medium text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-full border border-border">
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              v1.0.0 Server Active
            </div>
          </div>
        </header>

        {/* Page Content Scroll Area */}
        <div className="flex-1 overflow-auto relative">
          {/* Abstract background mesh */}
          <div className="absolute inset-0 pointer-events-none opacity-[0.03] mix-blend-screen z-0" 
               style={{ backgroundImage: `url(${import.meta.env.BASE_URL}images/abstract-mesh.png)`, backgroundSize: 'cover', backgroundPosition: 'center' }} />
          
          <div className="p-6 md:p-8 max-w-7xl mx-auto relative z-10">
            <motion.div
              key={location}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {children}
            </motion.div>
          </div>
        </div>
      </main>

      {/* Auth Dialog */}
      <Dialog open={isAuthDialogOpen} onOpenChange={setIsAuthDialogOpen}>
        <DialogContent className="sm:max-w-md glass-panel border-white/10">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 font-display text-xl">
              <Key className="w-5 h-5 text-primary" />
              Admin Authentication
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Provide your X-Admin-Key to manage API keys and internal server settings.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              type="password"
              placeholder="Enter Admin Key..."
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              className="bg-black/50 border-white/10 focus:border-primary text-white"
            />
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setIsAuthDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleSaveKey} className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/20">
              Save Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
