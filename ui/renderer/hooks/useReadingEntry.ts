import { useEffect, useState, type RefObject } from "react";

/** The key of the entry being read: the topmost [data-key] element in the upper part of the page. */
export function useReadingEntry(page: RefObject<HTMLElement | null>, deps: unknown[]): string | null {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    const element = page.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (items) => {
        const visible = items.filter((i) => i.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.getAttribute("data-key"));
      },
      { root: element, rootMargin: "0px 0px -60% 0px" },
    );
    element.querySelectorAll("[data-key]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return active;
}
