/**
 * Shared MaxCore (model server) connection config — single source of truth so
 * the URL and API key are not duplicated across modules.
 *
 * `ADMIN_KEY` should be supplied via the environment. The fallback below is the
 * model server's built-in localhost key for same-container IPC; it exists only
 * so local development works out of the box and matches the long-standing
 * default the rest of the api-server already relies on. Set `ADMIN_KEY` in the
 * environment to override it in any shared/deployed setting.
 */
export const MAXCORE_URL = `http://localhost:${process.env.MODEL_API_PORT || "9878"}`;

export const MAXCORE_API_KEY =
  process.env.ADMIN_KEY ||
  "mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc";
