import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8080,
    strictPort: true,
    host: true, // listen on all network interfaces, not just localhost, so it's reachable over LAN
  },
});
