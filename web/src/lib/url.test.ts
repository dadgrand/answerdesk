/** Hash-routing parameter helpers (deep links inside `#/console?...`). */

import { describe, expect, it } from "vitest";

import { urlParam } from "./url";

const location = (search: string, hash: string) => ({ search, hash });

describe("urlParam", () => {
  it("reads a normal query string", () => {
    expect(urlParam("mock", location("?mock=1", ""))).toBe("1");
  });

  it("reads parameters inside the hash route", () => {
    expect(urlParam("conversation", location("", "#/console?conversation=c-1042"))).toBe("c-1042");
  });

  it("prefers the search string when both are present", () => {
    expect(urlParam("session", location("?session=a", "#/console?session=b"))).toBe("a");
  });

  it("returns null when the parameter is absent or the hash has no query", () => {
    expect(urlParam("x", location("", "#/analytics"))).toBeNull();
    expect(urlParam("x", location("", ""))).toBeNull();
  });

  it("handles several parameters and URL encoding", () => {
    expect(urlParam("session", location("", "#/console?conversation=c-1&session=returning%20one"))).toBe(
      "returning one",
    );
  });
});
