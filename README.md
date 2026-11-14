# Edify Coach

One study coach per student, running on [Maritime](https://maritime.sh). Edify's server gives every student who
turns it on their own coach on their own machine. The coach remembers that student under `/data`, writes a revision
pack after each lecture, plans the next day every night (a Maritime schedule wakes it at 21:30 IST), and answers
questions with everything it knows about them.

## Contract (what Maritime expects)

| Route | What it does |
| --- | --- |
| `GET /health` | `{"ok": true}` |
| `GET /schedules` | The nightly review: Maritime wakes the coach for it, even from sleep |
| `POST /chat` `{"message": "..."}` | One event from Edify (JSON in `message`) or a plain question. Reply: `{"response": "<JSON>"}` |

Events Edify sends: `sync` (profile, weak spots, mocks, lectures), `lecture_done` (a finished lecture),
`review` (plan the day), `revised` (a revision ticked off), `ask` (a question), `state` (read only).

## Run it on this PC

```bat
set COACH_DATA=%TEMP%\edify-coach
set LLM_API_KEY=your-groq-key
python server.py
```

Then put `MARITIME_COACH_URL=http://127.0.0.1:8787` in `backend\.env` and restart Edify.

## Put it on Maritime

1. This folder is its own repo: push it to GitHub (the Dockerfile must be at the repo root).
2. In `backend\.env`: `MARITIME_API_KEY=mk_...` (from `maritime keys create --name edify --json`) and
   `MARITIME_COACH_REPO=https://github.com/<you>/edify-coach`.
3. Restart Edify. Each student who turns the coach on gets their own agent (`POST /api/v1/end-users/{id}/agents`).
