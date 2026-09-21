"use client";

import { useState } from "react";
import { postAssistantQuery, ApiError } from "@/lib/api-client";
import type { AssistantResponse } from "@/lib/types";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const EXAMPLE_QUESTIONS = [
  "Why is churn increasing among month-to-month customers?",
  "What does our retention playbook recommend for high-risk customers?",
  "What are the biggest risk factors for churn?",
];

export default function AssistantPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AssistantResponse | null>(null);

  function ask(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    postAssistantQuery(q.trim())
      .then(setResult)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "Request failed."))
      .finally(() => setLoading(false));
  }

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        AI Decision Assistant
      </h1>
      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
        Evidence-grounded answers over this platform&apos;s own data, models, and business documentation - not a
        general-purpose chatbot. Every number traces back to a tool result shown below the answer.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(query);
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={EXAMPLE_QUESTIONS[0]}
          className="flex-1 rounded-md border px-3 py-2 text-sm"
          style={{ borderColor: "var(--border)", background: "var(--surface-card)", color: "var(--text-primary)" }}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="rounded-md px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          style={{ background: "var(--series-1)" }}
        >
          Analyze
        </button>
      </form>
      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
        Try:{" "}
        {EXAMPLE_QUESTIONS.map((q, i) => (
          <span key={q}>
            <button type="button" onClick={() => { setQuery(q); ask(q); }} className="underline" style={{ color: "var(--series-1)" }}>
              &quot;{q}&quot;
            </button>
            {i < EXAMPLE_QUESTIONS.length - 1 && " · "}
          </span>
        ))}
      </p>

      {loading && <LoadingState label="Routing, gathering evidence, and generating an answer..." />}
      {error && <ErrorState message={error} />}

      {result && (
        <div className="space-y-4 rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
          <p className="font-medium" style={{ color: "var(--text-primary)" }}>{result.answer}</p>

          <div className="flex flex-wrap gap-4 text-xs" style={{ color: "var(--text-muted)" }}>
            <span>Route: <code>{result.route ?? "n/a"}</code></span>
            <span>Tools used: {result.tools_used.length > 0 ? result.tools_used.join(", ") : "none"}</span>
            {result.latency_ms && <span>{(result.latency_ms / 1000).toFixed(1)}s</span>}
            {result.confidence !== null && <span>Confidence: {((result.confidence ?? 0) * 100).toFixed(0)}%</span>}
          </div>

          {result.citations.length > 0 && (
            <div>
              <h3 className="mb-1 text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>Citations</h3>
              <ul className="list-inside list-disc text-xs" style={{ color: "var(--text-secondary)" }}>
                {result.citations.map((c, i) => (
                  <li key={i}>{c.label}</li>
                ))}
              </ul>
            </div>
          )}

          {result.evidence.length > 0 && (
            <details>
              <summary className="cursor-pointer text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
                Evidence ({result.evidence.length})
              </summary>
              <ul className="mt-2 space-y-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                {result.evidence.map((e, i) => (
                  <li key={i} className="rounded border p-2" style={{ borderColor: "var(--border)" }}>
                    <div className="font-medium">{e.claim}</div>
                    <div style={{ color: "var(--text-muted)" }}>
                      {e.source} - {e.value}
                    </div>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
