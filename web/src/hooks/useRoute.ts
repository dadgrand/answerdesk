/**
 * Hash routing: `#/console` (default) and `#/analytics`.
 *
 * A 40-line router instead of a dependency — the console has exactly two screens and the
 * URL is the only state that has to survive a reload (so a screenshot can deep-link).
 */

import { useCallback, useEffect, useState } from "react";

export type Route = "console" | "analytics";

export function parseRoute(hash: string): Route {
  const clean = hash.replace(/^#\/?/, "").split("?")[0];
  return clean === "analytics" ? "analytics" : "console";
}

export function useRoute(): { route: Route; navigate: (route: Route) => void } {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));

  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((next: Route) => {
    window.location.hash = next === "console" ? "/console" : "/analytics";
    setRoute(next);
  }, []);

  return { route, navigate };
}
