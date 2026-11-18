# Parsing Arena frontend

Next.js 16 app (App Router, Cache Components, Tailwind CSS 4). It talks to the FastAPI backend
through the `/api/*` rewrite in `next.config.ts`; set `BACKEND_URL` if the backend is not at
`http://127.0.0.1:8000`.

```bash
npm install
npm run dev     # http://localhost:3000
npm run lint
npm run build
```

Routes: `/` (landing page and model table), `/arena` (upload, run settings, history) and
`/experiments/[id]` (live progress and side-by-side results). See the root `README.md` for the
full setup.
