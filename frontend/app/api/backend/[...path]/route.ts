// Server-side proxy to the FastAPI backend. The browser calls
// /api/backend/<path>; this handler forwards to API_URL with the API key
// attached, so the key (API_KEY) never reaches the browser bundle and the
// backend needs no CORS configuration for the frontend.
//
// Only the endpoints the UI actually uses are forwarded - the key must not
// turn this route into an open door to the rest of the API.

const ALLOWED_PREFIXES = [
  "overview/",
  "at-risk",
  "customers",
  "outreach-draft/",
  "model-history",
  "pipeline-status",
  "assistant/query",
];

function isAllowed(path: string): boolean {
  return ALLOWED_PREFIXES.some((prefix) => path === prefix || path.startsWith(prefix));
}

async function forward(request: Request, segments: string[]): Promise<Response> {
  const path = segments.map(encodeURIComponent).join("/");
  if (!isAllowed(path)) {
    return Response.json({ detail: "not found" }, { status: 404 });
  }

  const apiUrl = process.env.API_URL ?? "http://localhost:8000";
  const search = new URL(request.url).search;
  const headers = new Headers({ "Content-Type": "application/json" });
  if (process.env.API_KEY) headers.set("X-API-Key", process.env.API_KEY);
  const clientIp = request.headers.get("x-forwarded-for");
  if (clientIp) headers.set("X-Forwarded-For", clientIp);

  try {
    const upstream = await fetch(`${apiUrl}/${path}${search}`, {
      method: request.method,
      headers,
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
    });
    const responseHeaders = new Headers({
      "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
    });
    const retryAfter = upstream.headers.get("Retry-After");
    if (retryAfter) responseHeaders.set("Retry-After", retryAfter);
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return Response.json({ detail: "backend unreachable" }, { status: 502 });
  }
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, { params }: Context) {
  return forward(request, (await params).path);
}

export async function POST(request: Request, { params }: Context) {
  return forward(request, (await params).path);
}
