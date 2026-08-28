import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No CDN, no external font host, no third-party map provider. Vehicle movement
// data for a whole country is a national security asset; the console that
// displays it must not phone anywhere out.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: true },
});
