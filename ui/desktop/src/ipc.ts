// Typed IPC: handlers and events must match the contract the page is written against.
import type { EventChannels, InvokeChannels } from "@agentomics/contract";
import { ipcMain, type IpcMainInvokeEvent, type WebContents } from "electron";

type Handler<C extends keyof InvokeChannels> = (
  event: IpcMainInvokeEvent,
  ...args: InvokeChannels[C]["args"]
) => InvokeChannels[C]["result"] | Promise<InvokeChannels[C]["result"]>;

/** The arguments are typed for convenience only: they come from the page, so handlers still check them. */
export function handle<C extends keyof InvokeChannels>(channel: C, handler: Handler<C>): void {
  ipcMain.handle(channel, handler as (event: IpcMainInvokeEvent, ...args: any[]) => unknown);
}

export function send<C extends keyof EventChannels>(contents: WebContents, channel: C, payload: EventChannels[C]): void {
  contents.send(channel, payload);
}
