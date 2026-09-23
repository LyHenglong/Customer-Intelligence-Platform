"use client";

import { useEffect, useState } from "react";

/**
 * Scoring the full customer population can take tens of seconds on a cold
 * cache. Static text sat there unchanged for that long reads as a frozen
 * page, so this shows motion plus an elapsed counter once the wait stops
 * being instant - the difference between "broken" and "working on it".
 */
export default function LoadingState({ label = "Loading..." }: { label?: string }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div
      className="flex items-center justify-center gap-3 rounded-lg border p-12"
      style={{ borderColor: "var(--border)" }}
      role="status"
      aria-live="polite"
    >
      <span
        aria-hidden="true"
        className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-transparent"
        style={{ borderTopColor: "var(--series-1)", borderRightColor: "var(--series-1)" }}
      />
      <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
        {label}
        {seconds >= 3 && (
          <span style={{ color: "var(--text-muted)" }}> ({seconds}s)</span>
        )}
      </span>
    </div>
  );
}
