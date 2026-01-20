import { api } from "./api";
import axios from "axios";

export async function checkBackendHealth() {
  // Primary attempt using configured baseURL (defaults to '/api/')
  try {
    const res = await api.get("health", { timeout: 5000 });
    if (res && res.status === 200 && res.data && res.data.status === "ok") {
      return true;
    }
  } catch (_) {
    // fall through to fallback
  }

  // Fallback for cases where the frontend is served without a proxy (e.g., opening build/index.html)
  // Allows overriding via REACT_APP_API_BASE_FALLBACK; defaults to localhost backend
  const FALLBACK_BASE = (process.env.REACT_APP_API_BASE_FALLBACK || "http://localhost:5000/api/")
    .replace(/\/?$/, "/");
  try {
    const res2 = await axios.get(`${FALLBACK_BASE}health`, { timeout: 5000 });
    return !!(res2 && res2.status === 200 && res2.data && res2.data.status === "ok");
  } catch (_) {
    return false;
  }
}

// Poll health periodically and invoke a callback with status
export function startHealthPolling(onStatus, intervalMs = 15000) {
  let timer = null;
  const poll = async () => {
    const ok = await checkBackendHealth();
    try { onStatus(ok); } catch (_) {}
  };
  // initial
  poll();
  timer = setInterval(poll, intervalMs);
  return () => {
    if (timer) clearInterval(timer);
  };
}
