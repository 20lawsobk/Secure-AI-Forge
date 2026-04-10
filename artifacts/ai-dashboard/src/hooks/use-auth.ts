import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  adminKey: string;
  setAdminKey: (key: string) => void;
  clearAdminKey: () => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      adminKey: "",
      setAdminKey: (key) => set({ adminKey: key }),
      clearAdminKey: () => set({ adminKey: "" }),
    }),
    {
      name: "ai-dashboard-auth",
    },
  ),
);

export function getAuthHeaders() {
  const state = useAuth.getState();
  return state.adminKey ? { "X-Admin-Key": state.adminKey } : {};
}
