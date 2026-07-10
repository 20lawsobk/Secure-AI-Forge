import { lazy, Suspense } from "react";
import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ErrorBoundary } from "@/components/error-boundary";
import { Layout } from "@/components/layout";
import { Skeleton } from "@/components/ui/skeleton";

const Dashboard = lazy(() => import("@/pages/dashboard"));
const ApiKeys = lazy(() => import("@/pages/api-keys"));
const Training = lazy(() => import("@/pages/training"));
const GpuStatus = lazy(() => import("@/pages/gpu"));
const ContentGenerator = lazy(() => import("@/pages/content"));
const ModelStatus = lazy(() => import("@/pages/model"));
const VideoStudio = lazy(() => import("@/pages/video-studio"));
const ArtistSettings = lazy(() => import("@/pages/artist-settings"));
const NotFound = lazy(() => import("@/pages/not-found"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30_000),
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
    mutations: {
      retry: 0,
    },
  },
});

function PageLoader() {
  return (
    <div className="space-y-6 p-2">
      <Skeleton className="h-10 w-56 bg-white/5" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-36 w-full rounded-2xl bg-white/5" />
        ))}
      </div>
      <Skeleton className="h-64 w-full rounded-2xl bg-white/5" />
    </div>
  );
}

function Router() {
  return (
    <Layout>
      <Suspense fallback={<PageLoader />}>
        <Switch>
          <Route path="/">
            <ErrorBoundary fallbackLabel="Dashboard error">
              <Dashboard />
            </ErrorBoundary>
          </Route>
          <Route path="/api-keys">
            <ErrorBoundary fallbackLabel="API Keys error">
              <ApiKeys />
            </ErrorBoundary>
          </Route>
          <Route path="/training">
            <ErrorBoundary fallbackLabel="Training error">
              <Training />
            </ErrorBoundary>
          </Route>
          <Route path="/gpu">
            <ErrorBoundary fallbackLabel="GPU status error">
              <GpuStatus />
            </ErrorBoundary>
          </Route>
          <Route path="/content">
            <ErrorBoundary fallbackLabel="Content generator error">
              <ContentGenerator />
            </ErrorBoundary>
          </Route>
          <Route path="/model">
            <ErrorBoundary fallbackLabel="Model status error">
              <ModelStatus />
            </ErrorBoundary>
          </Route>
          <Route path="/video">
            <ErrorBoundary fallbackLabel="Video studio error">
              <VideoStudio />
            </ErrorBoundary>
          </Route>
          <Route path="/artist-settings">
            <ErrorBoundary fallbackLabel="Brand Voice settings error">
              <ArtistSettings />
            </ErrorBoundary>
          </Route>
          <Route component={NotFound} />
        </Switch>
      </Suspense>
    </Layout>
  );
}

function App() {
  return (
    <ErrorBoundary fallbackLabel="Application error — please refresh">
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
            <Router />
          </WouterRouter>
          <Toaster />
        </TooltipProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
