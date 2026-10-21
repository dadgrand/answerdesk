/**
 * Query parameters that work with hash routing.
 *
 * The console is a hash-routed SPA (`#/console`), so a deep link looks like
 * `#/console?conversation=c-1042&mock=1`: the parameters live *inside* the hash, where
 * `location.search` cannot see them. This helper checks the real search string first (a
 * server that serves the app with a normal query still works) and then the hash query.
 */

export function urlParam(name: string, location: Pick<Location, "search" | "hash"> = window.location): string | null {
  const search = new URLSearchParams(location.search);
  const direct = search.get(name);
  if (direct !== null) return direct;
  const hash = location.hash ?? "";
  const queryIndex = hash.indexOf("?");
  if (queryIndex === -1) return null;
  return new URLSearchParams(hash.slice(queryIndex + 1)).get(name);
}
