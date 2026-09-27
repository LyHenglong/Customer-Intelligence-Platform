# Frontend

The platform's UI: a Next.js (App Router) app over the FastAPI serving layer
(`src/api/`). Pages live in `app/`, shared components in `components/`, and
`lib/api-client.ts` has one typed function per API endpoint, with response
types in `lib/types.ts`.

## Running it

```bash
npm ci
cp .env.local.example .env.local   # API_URL (and API_KEY, if the API has API_KEYS set)
npm run dev                        # http://localhost:3000
```

The API must be running too (`make serve` or `make api` from the repo root).

## How it talks to the API

The browser never calls the API directly. It calls this app's own route,
`/api/backend/<path>` (`app/api/backend/[...path]/route.ts`), which forwards
the request to `API_URL` with the `X-API-Key` header added on the server. So:

- the API key never appears in the browser bundle;
- the API needs no CORS configuration;
- `API_URL`/`API_KEY` are read at runtime, so one built image works against
  any API;
- only the endpoints the UI uses are forwarded.

Setting `NEXT_PUBLIC_API_URL` switches to direct browser-to-API calls instead.
It is inlined at build time and only works while the API has no `API_KEYS`.

## Checks

```bash
npm run lint
npm run typecheck    # generates Next's route types, then tsc --noEmit
npm run build
npm test             # Playwright smoke tests against the production build
```

The smoke tests (`e2e/`) load every page with the API mocked from typed
fixtures (`e2e/mock-api.ts`), so they need no backend. They run in CI after
the build; locally, run `npm run build` first.
