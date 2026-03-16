import { Component, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error("[ErrorBoundary] Caught render error:", error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[40vh] gap-6 p-8">
          <div className="flex items-center gap-3 text-destructive">
            <AlertTriangle className="w-8 h-8" />
            <h2 className="text-xl font-display font-semibold">
              {this.props.fallbackLabel ?? "Something went wrong"}
            </h2>
          </div>
          {this.state.error && (
            <p className="text-sm text-muted-foreground max-w-md text-center font-mono">
              {this.state.error.message}
            </p>
          )}
          <Button variant="outline" onClick={this.handleReset} className="gap-2">
            <RefreshCw className="w-4 h-4" />
            Try again
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}
