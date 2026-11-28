/** A desktop call's error in the user's words, without Electron's "Error invoking remote method" prefix. */
export function invokeError(error: unknown): string {
  return String((error as Error)?.message ?? error).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
}
