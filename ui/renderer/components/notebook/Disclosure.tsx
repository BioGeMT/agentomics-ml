import { ChevronRight } from "lucide-react";

/** The chevron that turns when an iteration opens. */
export function Disclosure({ open }: { open: boolean }) {
  return <ChevronRight size={14} className={`disclosure ${open ? "open" : ""}`} aria-hidden />;
}
