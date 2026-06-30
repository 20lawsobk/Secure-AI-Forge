export const logger = {
  info: (...args: unknown[]): void => {
    console.log("[Info]", ...args);
  },
  warn: (...args: unknown[]): void => {
    console.warn("[Warn]", ...args);
  },
  error: (...args: unknown[]): void => {
    console.error("[Error]", ...args);
  },
  debug: (...args: unknown[]): void => {
    if (process.env.NODE_ENV === "development") {
      console.debug("[Debug]", ...args);
    }
  },
};
