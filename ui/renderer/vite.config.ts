// Builds the renderer: two pages (app/ is the lobby, run/ a run's notebook) to out/, which the desktop
// app serves through agentomics://. In development the dev server also answers /api/* (dev-api.ts).
import react from "@vitejs/plugin-react";
import path from "node:path";
import { defineConfig } from "vite";
import { devApi } from "./dev-api.ts";

const here = import.meta.dirname;

export default defineConfig({
  root: here,
  // Two separate pages, so /run/?… is never answered with the lobby.
  appType: "mpa",
  plugins: [react(), devApi()],
  resolve: {
    // The same paths as tsconfig.json; the more specific alias comes first.
    alias: [
      { find: /^@agentomics\/contract\/(.+)$/, replacement: path.resolve(here, "../contract/$1.ts") },
      { find: "@agentomics/contract", replacement: path.resolve(here, "../contract/index.ts") },
      { find: "@agentomics/server", replacement: path.resolve(here, "../server/index.ts") },
      { find: /^@\//, replacement: `${here}/` },
    ],
  },
  server: { host: "127.0.0.1", port: 3000, strictPort: true, fs: { allow: [path.resolve(here, "..")] } },
  build: {
    outDir: "out",
    emptyOutDir: true,
    rollupOptions: { input: { index: path.resolve(here, "index.html"), app: path.resolve(here, "app/index.html"), run: path.resolve(here, "run/index.html") } },
  },
});
