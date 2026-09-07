// 开发时将 API 交给本机 FastAPI；生产构建由 FastAPI 同源托管。
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
