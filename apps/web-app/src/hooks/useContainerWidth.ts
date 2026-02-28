import { useEffect, useRef, useState } from "react";

const DEFAULT_FALLBACK_WIDTH = 320;

export function useContainerWidth(fallbackWidth = DEFAULT_FALLBACK_WIDTH) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(fallbackWidth);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }

    if (typeof ResizeObserver === "undefined") {
      setWidth(Math.floor(element.getBoundingClientRect().width) || fallbackWidth);
      return;
    }

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const measuredWidth = Math.floor(entry.contentRect.width);
        if (measuredWidth > 0) {
          setWidth(measuredWidth);
        }
      }
    });

    resizeObserver.observe(element);
    return () => resizeObserver.disconnect();
  }, [fallbackWidth]);

  return { containerRef, width };
}
