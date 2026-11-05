import { useEffect, useState, type RefObject } from "react";
import type { Run } from "@/model/run";

/**
 * Keeps a live run's newest entry in view while the reader is at the bottom of the page. A replay is
 * not followed: it opens at the top, like any finished document.
 */
export function useFollow(page: RefObject<HTMLElement | null>, run: Run, live: boolean) {
  const [following, setFollowing] = useState(true);

  useEffect(() => {
    if (live && run.status === "running" && following) page.current?.scrollTo({ top: page.current.scrollHeight });
  }, [page, run, live, following]);

  /** For the page's onScroll: following resumes once the reader scrolls back to the bottom. */
  function onScroll() {
    const element = page.current;
    if (!element) return;
    setFollowing(element.scrollHeight - element.scrollTop - element.clientHeight < 80);
  }

  function jumpToLive() {
    setFollowing(true);
    page.current?.scrollTo({ top: page.current.scrollHeight, behavior: "smooth" });
  }

  return { following, stopFollowing: () => setFollowing(false), onScroll, jumpToLive };
}
