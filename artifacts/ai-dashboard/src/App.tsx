import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";

import { Layout } from "@/components/layout";
import Dashboard from "@/pages/dashboard";
import ApiKeys from "@/pages/api-keys";
import Training from "@/pages/training";
import GpuStatus from "@/pages/gpu";
import ContentGenerator from "@/pages/content";
import ModelStatus from "@/pages/model";
import VideoStudio from "@/pages/video-studio";
import { ErrorBoundary } from "@/components/error-boundary";

// Global config for React Query — 2 retries with exponential back-off
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30_000),
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0,
    },
  },
});

function Router() {
  return (
    <Layout>
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
        <Route component={NotFound} />
      </Switch>
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
