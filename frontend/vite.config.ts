import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  resolve: { dedupe: ["three"] },
  build: {
    license: { fileName: "assets/THIRD_PARTY_LICENSES.md" },
  },
  server: {
    port: 5178,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000" },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
