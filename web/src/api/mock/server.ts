/** MSW node server for vitest: the same handlers the browser worker uses. */

import { setupServer } from "msw/node";

import { mockHandlers } from "./handlers";

export const mockServer = setupServer(...mockHandlers);
