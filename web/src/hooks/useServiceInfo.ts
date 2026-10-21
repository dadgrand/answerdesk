/**
 * Service identity + health for the top bar.
 *
 * `/v1/info` is the non-secret identity the service exposes (profile, model, corpus and
 * index versions); `/healthz` says whether the dependencies are healthy. The console does
 * not hide a broken service: a failed probe renders an actionable banner instead of empty
 * panels.
 */

import { useEffect, useState } from "react";

import { fetchHealth, fetchInfo } from "../api/client";
import type { HealthResponse, InfoResponse } from "../api/types";

export type ServiceState =
  | { status: "loading" }
  | { status: "ready"; info: InfoResponse; health: HealthResponse | null }
  | { status: "error"; message: string };

export function useServiceInfo(pollMs = 30_000): ServiceState {
  const [state, setState] = useState<ServiceState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const load = async () => {
      try {
        const info = await fetchInfo();
        let health: HealthResponse | null = null;
        try {
          health = await fetchHealth();
        } catch {
          health = null;
        }
        if (!cancelled) setState({ status: "ready", info, health });
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
      if (!cancelled && pollMs > 0) timer = window.setTimeout(load, pollMs);
    };

    void load();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [pollMs]);

  return state;
}
