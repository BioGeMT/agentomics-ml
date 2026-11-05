// Compiles the main process and the preload into dist/. Type checking is separate: npm run typecheck.
import { build } from "esbuild";

// One tsconfig for every file, so ../server and ../contract resolve the same @agentomics/* paths.
const common = { tsconfig: "tsconfig.json", bundle: true, platform: "node", format: "cjs", target: "node22", external: ["electron"], sourcemap: "linked", logLevel: "warning" };

await Promise.all([
  // The main process, with ../server bundled in.
  build({ ...common, entryPoints: ["src/main.ts"], outfile: "dist/main.js" }),
  // The preload runs sandboxed, where only `electron` can be required: everything else is bundled.
  build({ ...common, entryPoints: ["src/preload.ts"], outfile: "dist/preload.js" }),
]);
