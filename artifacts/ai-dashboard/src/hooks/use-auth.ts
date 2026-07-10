import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  adminKey: string;
  setAdminKey: (key: string) => void;
  clearAdminKey: () => void;
  // The active artist's Brand Voice profile id — persisted so the Settings
  // screen and any generation forms that want to pass `artistProfileId` all
  // agree on which profile is "current" without re-typing it everywhere.
  artistProfileId: string;
  setArtistProfileId: (id: string) => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      adminKey: "",
      setAdminKey: (key) => set({ adminKey: key }),
      clearAdminKey: () => set({ adminKey: "" }),
      artistProfileId: "",
      setArtistProfileId: (id) => set({ artistProfileId: id }),
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
