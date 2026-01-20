import axios from "axios";

// Prefer env var, else use relative '/api' so CRA dev server proxy works
// Ensure trailing slash for consistent URL joining
const API_BASE = (process.env.REACT_APP_API_BASE || "/api")
  .replace(/\/?$/, "/");
// Only enable verbose backend debug when explicitly requested
const DEFAULT_DEBUG = (process.env.REACT_APP_API_DEBUG || "").toLowerCase() === "true";

export const api = axios.create({
  baseURL: API_BASE,
});

// Automatically append ?debug=true for manual predict calls when DEFAULT_DEBUG is on
api.interceptors.request.use((config) => {
  try {
    if (!config || !config.url) return config;
    if (!DEFAULT_DEBUG) return config;
    // Normalize URL without leading slash to ensure baseURL applies
    let url = String(config.url);
    const isManualPredict = url.endsWith("manual/predict") || url.endsWith("/manual/predict");
    if (isManualPredict) {
      config.params = { ...(config.params || {}), debug: "true" };
    }
  } catch (_) {
    // no-op
  }
  return config;
});
