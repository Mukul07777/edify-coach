"""Edify Coach: one study coach per student, running on Maritime (https://maritime.sh).

Each student gets their own coach on their own machine. The coach keeps its memory of that student under /data
(what they studied, what they keep getting wrong, what it planned for them), so it gets to know them over weeks
instead of starting from nothing on every call. It does three jobs:

* after every lecture, a revision pack: the few ideas that matter, the formulas, practice questions with answers,
  and the days to revise it again (1, 3 and 7 days on: spaced repetition);
* every night (a Maritime schedule wakes it, even while it sleeps), a review: tomorrow's plan, built from the
  revisions that fall due, the open weak spots and the last mock tests, with a short note to the student;
* questions, answered with everything it remembers about the student.

The Maritime container contract: listen on 0.0.0.0:$PORT, answer GET /health, answer POST /chat {"message"}
within 30 seconds, keep state under /data. Edify's server sends events as JSON inside "message"; anything that is
not JSON is treated as a question. Standard library only, so the image builds in seconds.
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VERSION = "1.0.0"
IST = timezone(timedelta(hours=5, minutes=30))
DATA = Path(os.environ.get("COACH_DATA") or "/data") / "edify-coach"
MEMORY = DATA / "memory.json"  # one student per machine on Maritime
LLM_TIMEOUT = 18  # seconds: Maritime gives a chat turn 30
REVISE_AFTER = (1, 3, 7)  # days after a lecture to revise it again
NIGHTLY = {"id": "nightly-review", "cron": "30 21 * * *", "tz": "Asia/Kolkata",
           "prompt": json.dumps({"type": "review", "reason": "nightly"}), "enabled": True}

_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------------- memory
def _blank() -> dict[str, Any]:
    return {"student": {}, "weak": [], "mocks": [], "sessions": [], "lectures": [], "plan": None,
            "questions": [], "reviews": 0, "last_review": None, "created_at": _now()}


def _now() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _today() -> date:
    return datetime.now(IST).date()


def _file(student: str | None) -> Path:
    """On Maritime a machine holds one student. One coach serving several (a PC standing in for Maritime) keeps
    each student's memory apart, by the id Edify sends."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", student or "")[:60]
    return DATA / "students" / safe / "memory.json" if safe else MEMORY


def load(student: str | None = None) -> dict[str, Any]:
    try:
        return {**_blank(), **json.loads(_file(student).read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return _blank()


def save(mem: dict[str, Any], student: str | None = None) -> None:
    path = _file(student)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # all or nothing: a reboot mid-write never leaves half a memory


# ---------------------------------------------------------------------------------------------------- the model
def _llm_config() -> tuple[str, str, str] | None:
    """Maritime's own LLM when it is switched on for the agent (OPENAI_*), else the key Edify passed in (LLM_*)."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    if not key:
        return None
    base = (os.environ.get("OPENAI_BASE_URL") if os.environ.get("OPENAI_API_KEY") else os.environ.get("LLM_BASE_URL")) \
        or "https://api.groq.com/openai/v1"
    model = os.environ.get("COACH_MODEL") or ("openai/gpt-oss-20b" if "groq" in base else "gpt-4o-mini")
    return key, base.rstrip("/"), model


def ask_llm(system: str, user: str, max_tokens: int = 1400, salvage: str | None = None) -> dict[str, Any] | None:
    """One JSON answer from the model, or None (no key, slow, or unreadable): every caller has a fallback.
    ``salvage``: the one text field to rescue from a reply cut short (reasoning models spend tokens before answering)."""
    cfg = _llm_config()
    if not cfg:
        return None
    key, base, model = cfg
    body = json.dumps({"model": model, "temperature": 0.3, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system + "\nReply with one JSON object only."},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": f"edify-coach/{VERSION}"})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, OSError):
        return None
    match = re.search(r"\{.*\}", text, re.S)
    try:
        if match:
            return json.loads(match.group(0))
    except ValueError:
        pass
    if salvage:  # {"answer": "...  (cut off): keep what arrived; plain prose: take it as the answer
        cut = re.search(r'"%s"\s*:\s*"(.*)' % salvage, text, re.S)
        prose = cut.group(1).rstrip('"} \n') if cut else text.strip()
        try:
            prose = json.loads(f'"{prose}"') if cut else prose
        except ValueError:
            prose = prose.replace("\\n", "\n")
        if len(prose) > 20:
            return {salvage: prose}
    return None


def _lang(mem: dict[str, Any]) -> str:
    return LANGS.get((mem.get("student") or {}).get("language") or "", "English")


# the student's language, said so the model writes every sentence in it (technical terms and formulas stay English)
LANGS = {
    "hinglish": "Hinglish (Hindi in Roman letters, mixed with English)",
    "hi": "Hindi (Devanagari)",
    "mr": "Marathi (Devanagari script; Marathi, not Hindi: आहे, नाही, आणि)",
    "bn": "Bengali (Bengali script)",
    "ta": "Tamil (Tamil script)",
    "te": "Telugu (Telugu script)",
    "gu": "Gujarati (Gujarati script)",
}
TERMS = ", every sentence, with technical terms, formulas and units kept in English as Indian classrooms say them"


def _voice(mem: dict[str, Any]) -> str:
    s = mem.get("student") or {}
    lang = _lang(mem)
    return (f"You are Edify Coach, the personal study coach of {s.get('name') or 'a student'}, preparing for "
            f"{s.get('exam') or 'their exams'}. Be warm, specific and brief, like a senior who cleared the exam. "
            f"Write in {lang}{TERMS if lang != 'English' else ''}. Never invent marks, dates or facts you were not given.")


# ---------------------------------------------------------------------------------------------------- the jobs
def sync(mem: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Edify's latest picture of the student: profile, open weak spots, recent mock tests and lectures."""
    for key in ("student", "weak", "mocks", "sessions", "gaps"):
        if key in event:
            mem[key] = event[key]
    mem["synced_at"] = _now()
    return {"ok": True, "synced_at": mem["synced_at"]}


def _fallback_pack(lecture: dict[str, Any]) -> dict[str, Any]:
    summary = lecture.get("summary_md") or ""
    lines = [ln.strip(" -*#\t") for ln in summary.splitlines() if len(ln.strip(" -*#\t")) > 2]
    formulas = [ln for ln in lines if "=" in ln][:6]  # short is fine for a formula: "L = Iω"
    points = [ln for ln in lines if "=" not in ln and len(ln) > 12][:5]
    return {"headline": f"Revise {lecture.get('title') or 'this lecture'} while it is fresh.",
            "key_points": points or ["Read your notes from this lecture once more, slowly."],
            "formulas": formulas,
            "practice": [{"q": f"Without looking, write down the main idea of {lecture.get('title') or 'the lecture'} and one example.",
                          "answer": "Check it against your notes."}],
            "source": "offline"}


def _texts(value: Any, limit: int) -> list[str]:
    return [str(v).strip() for v in value if str(v).strip()][:limit] if isinstance(value, list) else []


def _clean_pack(pack: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """The model's pack in the exact shape the app reads: a model sometimes drops a field, returns a number where
    text belongs, or a question without its answer. Anything missing is filled from the lecture itself."""
    practice = [{"q": str(p.get("q") or p.get("question") or "").strip(), "answer": str(p.get("answer") or "").strip()}
                for p in pack.get("practice") or [] if isinstance(p, dict)]
    practice = [p for p in practice if p["q"]][:5]
    return {
        "headline": str(pack.get("headline") or fallback["headline"]).strip(),
        "key_points": _texts(pack.get("key_points"), 6) or fallback["key_points"],
        "formulas": _texts(pack.get("formulas"), 8) or fallback["formulas"],
        "practice": practice or fallback["practice"],
        "link": str(pack.get("link") or "").strip(),
        "source": "ai",
    }


def lecture_done(mem: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """A revision pack for the lecture just finished, remembered with the days to revise it again."""
    lec = event.get("lecture") or {}
    lid = str(lec.get("id") or lec.get("title") or len(mem["lectures"]))
    pack = ask_llm(
        _voice(mem) + " You turn a finished lecture into a revision pack.",
        "Lecture: " + json.dumps({k: lec.get(k) for k in ("title", "subject", "summary_md")}, ensure_ascii=False)[:6000]
        + "\nThe student's open weak spots: " + json.dumps(mem.get("weak", [])[:8], ensure_ascii=False)
        + '\nReturn {"headline": one line on what to take away, "key_points": 3-5 short points, '
          '"formulas": the formulas to remember (plain text, may be empty), '
          '"practice": 3 exam-style questions as [{"q", "answer"}], '
          '"link": one line tying this lecture to a weak spot when one is related, else ""}.')
    if not pack or not isinstance(pack.get("key_points"), list):
        pack = _fallback_pack(lec)
    else:
        pack = _clean_pack(pack, _fallback_pack(lec))
    today = _today()
    pack.update({"lecture_id": lid, "title": lec.get("title") or "Lecture", "subject": lec.get("subject"),
                 "made_at": _now(), "revise_on": [(today + timedelta(days=d)).isoformat() for d in REVISE_AFTER],
                 "revised": []})
    mem["lectures"] = [x for x in mem["lectures"] if x.get("lecture_id") != lid][-40:] + [pack]
    return {"pack": pack}


def _due(mem: dict[str, Any], day: date) -> list[dict[str, Any]]:
    out = []
    for lec in mem["lectures"]:
        due = [d for d in lec.get("revise_on", []) if d <= day.isoformat() and d not in lec.get("revised", [])]
        if due:
            out.append({"title": lec["title"], "subject": lec.get("subject"), "lecture_id": lec["lecture_id"], "round": len(lec.get("revised", [])) + 1})
    return out


def review(mem: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Tomorrow's plan (today's, when asked during the day): revisions due, weak spots, then the last mock's gaps."""
    nightly = event.get("reason") == "nightly"
    day = _today() + timedelta(days=1) if nightly else _today()
    if nightly and mem.get("plan") and mem["plan"].get("for") == day.isoformat():
        return {"plan": mem["plan"], "skipped": "already planned"}  # the schedule fired twice: one plan a night
    items: list[dict[str, Any]] = []
    # a root gap first: the topic above it can't hold until the idea underneath does
    for g in (mem.get("gaps") or [])[:2]:
        where = f" (Class {g['cls']})" if g.get("cls") else ""
        items.append({"kind": "root", "title": f"Fix the root: {g.get('root')}", "minutes": 10,
                      "why": f"It's why {g.get('stuck_on')} isn't clicking; from {g.get('chapter')}{where}", "trace_id": g.get("trace_id")})
    for d in _due(mem, day)[:4]:
        items.append({"kind": "revise", "title": f"Revise {d['title']}", "why": f"Spaced revision, round {d['round']} of 3",
                      "minutes": 20, "lecture_id": d["lecture_id"]})
    for w in sorted(mem.get("weak", []), key=lambda w: -(w.get("misses") or 0))[:3]:
        items.append({"kind": "weak", "title": f"Fix: {w.get('topic') or w.get('chapter')}",
                      "why": f"Lost {w.get('misses', 1)} question{'s' if (w.get('misses') or 1) != 1 else ''} on it in tests", "minutes": 30})
    mocks = [m for m in mem.get("mocks", []) if m.get("max_score")]
    if mocks:
        last = mocks[0]
        pct = round(100 * (last.get("score") or 0) / last["max_score"])
        items.append({"kind": "mock", "title": f"Go over the mistakes in {last.get('title') or 'your last mock'}",
                      "why": f"You scored {pct}%: every wrong answer reviewed is a mark next time", "minutes": 25})
    fresh = [x for x in mem["lectures"] if (x.get("made_at") or "")[:10] == _today().isoformat()]
    if not items and fresh:  # nothing due yet, but a lecture today: a first look back while it is fresh
        items.append({"kind": "revise", "title": f"Look over today's pack: {fresh[-1]['title']}",
                      "why": "Ten minutes now beats an hour next week", "minutes": 10})
    if not items:
        items.append({"kind": "start", "title": "Watch one lecture with Edify", "why": "Your coach learns from every lecture you finish", "minutes": 45})

    note = ask_llm(
        _voice(mem) + " You write the student's plan note for the day.",
        "Plan items: " + json.dumps(items, ensure_ascii=False)
        + "\nRecent lectures: " + json.dumps([{"title": x["title"], "made_at": x["made_at"]} for x in mem["lectures"][-5:]], ensure_ascii=False)
        + "\nRecent mocks: " + json.dumps(mem.get("mocks", [])[:3], ensure_ascii=False)
        + f'\nWrite in {_lang(mem)}. Return {{"headline": one encouraging, specific line (max 14 words), '
          '"nudges": 2-3 short, specific nudges about habits or gaps you can see in the data}.', max_tokens=1000) or {}
    plan = {"for": day.isoformat(), "made_at": _now(), "items": items,
            "minutes": sum(i["minutes"] for i in items),
            "headline": note.get("headline") or f"{len(items)} focused task{'s' if len(items) != 1 else ''} for {'tomorrow' if nightly else 'today'}.",
            "nudges": [n for n in (note.get("nudges") or []) if isinstance(n, str)][:3],
            "source": "ai" if note else "offline"}
    mem["plan"] = plan
    mem["reviews"] = mem.get("reviews", 0) + 1
    mem["last_review"] = _now()
    return {"plan": plan}


def revised(mem: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """The student ticked a revision off: the next round moves on."""
    for lec in mem["lectures"]:
        if lec["lecture_id"] == str(event.get("lecture_id")):
            due = [d for d in lec.get("revise_on", []) if d <= _today().isoformat() and d not in lec["revised"]]
            if due:
                lec["revised"].append(due[0])
            return {"ok": True, "lecture_id": lec["lecture_id"], "revised": lec["revised"]}
    return {"ok": False}


def answer(mem: dict[str, Any], question: str) -> dict[str, Any]:
    context = {"weak": mem.get("weak", [])[:8], "mocks": mem.get("mocks", [])[:3], "root_gaps": (mem.get("gaps") or [])[:4],
               "recent_lectures": [{"title": x["title"], "key_points": x.get("key_points", [])[:3]} for x in mem["lectures"][-4:]],
               "plan": (mem.get("plan") or {}).get("items")}
    out = ask_llm(_voice(mem) + " Answer the student's question using what you know about them.",
                  "What you know: " + json.dumps(context, ensure_ascii=False)[:5000] + "\nQuestion: " + question[:1500]
                  + f'\nReturn {{"answer": your answer in markdown, written in {_lang(mem)}, under 180 words, no title line}}.',
                  max_tokens=1600, salvage="answer")
    reply = (out or {}).get("answer") or "I can't reach my AI right now. Try again in a minute; your plan above still holds."
    mem["questions"] = (mem.get("questions", []) + [{"q": question[:300], "at": _now()}])[-30:]
    return {"answer": reply}


def state(mem: dict[str, Any]) -> dict[str, Any]:
    return {"plan": mem.get("plan"), "packs": list(reversed(mem["lectures"][-5:])), "due_today": _due(mem, _today()),
            "reviews": mem.get("reviews", 0), "last_review": mem.get("last_review"), "lectures_seen": len(mem["lectures"]),
            "since": mem.get("created_at"), "version": VERSION}


JOBS = {"sync": sync, "lecture_done": lecture_done, "review": review, "revised": revised}


def handle(message: str, source: str | None) -> dict[str, Any]:
    try:
        event = json.loads(message)
    except ValueError:
        event = None
    if not isinstance(event, dict):
        # a schedule's own prompt arrives as text; anything else is a question
        event = {"type": "review", "reason": "nightly"} if source == "scheduled" else {"type": "ask", "question": message}
    kind = event.get("type")
    with _lock:
        student = event.get("student_id")
        mem = load(student)
        if kind == "ask":
            out = answer(mem, str(event.get("question") or ""))
        elif kind in JOBS:
            out = JOBS[kind](mem, event)
        elif kind == "state":
            out = {}
        else:
            return {"error": f"unknown event {kind!r}"}
        if kind != "state":
            save(mem, student)
        out["state"] = state(mem)
    return out


# ---------------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = f"edify-coach/{VERSION}"

    def _send(self, code: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "version": VERSION})
        elif self.path.startswith("/schedules"):
            self._send(200, [NIGHTLY])  # Maritime wakes the coach for its nightly review, even from sleep
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/chat"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._send(400, {"error": "send JSON: {\"message\": \"...\"}"})
        result = handle(str(body.get("message") or ""), body.get("source"))
        # Maritime passes "response" through as the reply; Edify reads the JSON inside it
        self._send(200, {"response": json.dumps(result, ensure_ascii=False)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{_now()} {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    port = int(os.environ.get("PORT") or 8787)
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Edify Coach {VERSION} on 0.0.0.0:{port}, memory in {DATA}, AI {'on' if _llm_config() else 'off'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
