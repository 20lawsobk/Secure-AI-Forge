import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { setAuthHeaderProvider } from "@workspace/api-client-react";
import { useAuth } from "./hooks/use-auth";

setAuthHeaderProvider(() => {
  const { adminKey } = useAuth.getState();
  return adminKey
    ? { "X-Admin-Key": adminKey }
    : ({} as Record<string, string>);
});

createRoot(document.getElementById("root")!).render(<App />);
