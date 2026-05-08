import os
import re
import json
import asyncio
import time
from typing import List, Optional, Dict


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> / <thinking>…</thinking> thought blocks
    emitted by models like Gemma 4. Without this the parser counts TOPIC:
    headers inside the model's scratchpad."""
    if not text:
        return ""
    return re.sub(r'<(think|thinking|thought)\b[^>]*>.*?</\1>', '',
                  text, flags=re.DOTALL | re.IGNORECASE).strip()


# ── Lightweight answer-similarity prefilter ───────────────────────
# Words that don't disambiguate a quizbowl answer. After stripping these
# (plus articles/connectives), "P vs NP problem" and "P = NP" both
# reduce to {p, np} and we can accept without an LLM call.
_NOISE_WORDS = {
    # articles / connectives
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for",
    "vs", "versus", "is", "as",
    # quizbowl category labels — eponym is the answer, this is decoration
    "problem", "law", "theorem", "principle", "algorithm", "equation",
    "conjecture", "hypothesis", "function", "technique", "method",
    "effect", "model", "process",
}


def _normalize_tokens(s: str) -> set:
    """Lowercase → tokens, with operator chars (=, +, −, /) treated as
    whitespace and noise/category words dropped. Returns a set."""
    if not s:
        return set()
    s = re.sub(r"[=+\-/&]", " ", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)        # strip remaining punctuation
    return {t for t in s.split() if t and t not in _NOISE_WORDS}


def _quick_match(answer: str, guess: str) -> bool:
    """Conservative pre-judge: accepts when the guess clearly names the same
    thing as the answer. Two checks in order:
    1. Case-insensitive exact string match (catches pure capitalization diff).
    2. Normalized token-set equality (catches "P=NP" / "P vs NP problem",
       "Lenz" / "Lenz's law"). Anything fuzzier falls through to the LLM."""
    if answer.strip().lower() == guess.strip().lower():
        return True
    a, g = _normalize_tokens(answer), _normalize_tokens(guess)
    if not a or not g:
        return False
    return a == g


def _parse_answer_line(line: str):
    """Parse 'Primary [or alt1; or alt2]' → ('Primary', ['alt1', 'alt2']).

    Quizbowl convention puts accepted alternates in square brackets,
    separated by '; or' / 'or' / 'accept'. Plain answers without brackets
    return an empty alias list."""
    if not line:
        return "", []
    line = line.strip()
    m = re.match(r"^([^\[\]]+?)(?:\s*\[(.+?)\])?\s*$", line)
    if not m:
        return line, []
    primary = m.group(1).strip().strip(".,;")
    bracket = (m.group(2) or "").strip()
    if not bracket:
        return primary, []
    aliases = []
    for part in re.split(r"[;,]", bracket):
        part = part.strip()
        part = re.sub(r"^(or|accept|also accept|prompt on|prompt)\s+",
                      "", part, flags=re.IGNORECASE).strip()
        part = part.strip(".,;\"' ")
        if part and part.lower() != primary.lower():
            aliases.append(part)
    return primary, aliases

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Clients (both speak the OpenAI Chat Completions dialect)
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
# Gemini exposes an OpenAI-compatible endpoint, so the same SDK works.
gemini_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ── Models ────────────────────────────────────────────────────────
# Two-phase pipeline:
#  1) Bootstrap (until 3 clues ready): one writer call picks topic AND writes
#     the tossup, using fast Llama 70B. Lowest latency to first clue.
#  2) Steady state: architect picks topics in batches (Gemma 27B, smart but
#     fast); writer turns each topic+facts into a tossup. Writer model is
#     queue-depth-aware: fast Llama when shallow (≤ FAST_MODEL_THRESHOLD),
#     slow Gemini Flash Lite when deep — saves Groq quota.
MODEL_ARCHITECT   = "gemini-3.1-flash-lite"      # Google, non-thinking (Gemma 4 26B's reasoning loop was minutes per call)
MODEL_WRITER_FAST = "llama-3.3-70b-versatile"    # Groq, ~few seconds
MODEL_WRITER_SLOW = "gemini-3.1-flash-lite"      # Google, slower / lighter on Groq quota
MODEL_JUDGE       = "llama-3.1-8b-instant"       # Groq, sub-second validation

# ── Room password (WebSocket) ─────────────────────────────────────
# If set, clients must pass the same value as query param: /ws/{id}?token=...
# Leave empty for local dev with no gate.
API_SHARED_TOKEN = os.environ.get("API_SHARED_TOKEN", "").strip()

SYSTEM_PROMPT_ARCHITECT = """
Pick N Quizbowl topics in the subject. For each, list around 2 facts — ideally
they should NOT be common knowledge. Order doesn't matter; the writer arranges them.
Avoid any names in the "DO NOT pick" list.

### Example
TOPIC: Leo Tolstoy
FACTS:
- 1901 excommunication by the Russian Orthodox Holy Synod after publishing "Resurrection"
- Created Konstantin Levin, who proposes to Kitty by writing the first letters of words in chalk
- Russian author of War and Peace and Anna Karenina

### Output
Repeat the block N times. No preamble.

TOPIC: <name>
FACTS:
- <fact>
- <fact>
- <fact>
"""

SYSTEM_PROMPT_WRITER_PICK = """
You are an expert Quizbowl writer. Given a subject area, pick a SPECIFIC,
NAMED topic and write a complete pyramidal tossup. Used when speed matters.

### Selection
Specific named entities only (works, people, named laws/theorems, specific
concepts). If a "DO NOT pick" list is given, avoid those exactly AND avoid
anything that would be an alias of those (e.g. "Lev Tolstoy" if "Leo Tolstoy"
is on the list).

### Tossup Style (HS / NAQT level)
- 3-4 sentences, ~80-130 words.
- Pyramidal: most obscure clue first, most common last.
- Refer to the answer only as "this <category>".
- Concrete, factual clues only — specific names, dates, places, terms.
- End with: "For 10 points, name this <category> <giveaway description>."
- Never include the answer, alias, abbreviation, eponym, synonym, or translation
  (e.g. "printf debugging" leaks "print debugging"). No "also known as".

### Answer line
List every common alternate name in square brackets, separated by "; or".
This is what the moderator will accept as a correct buzz. Examples:
- "Mark Twain [or Samuel Clemens; or Samuel Langhorne Clemens]"
- "Aliens trick [or WQS binary search; or Lagrange optimization on convex DP]"
- "Knuth–Morris–Pratt algorithm [or KMP]"
If there are genuinely no alternates, just write the name with no brackets.

### Example
TOPIC: Leo Tolstoy
ANSWER: Leo Tolstoy [or Lev Nikolayevich Tolstoy; or Lev Tolstoy]
CLUE: A 1901 excommunication by the Russian Orthodox Holy Synod followed this man's publication of a novel in which he satirized the church through the prostitute Maslova. In a novella by this author, Praskovya Fedorovna is tormented by the three-day scream of a dying judge. He created Konstantin Levin, who proposes to Kitty Shcherbatskaya by writing the first letters of words in chalk. He opened another novel by declaring that "all happy families are alike; each unhappy family is unhappy in its own way." For 10 points, name this Russian author of War and Peace and Anna Karenina.

### Output (no preamble)
TOPIC: <name>
ANSWER: <answer>[ optional bracket of alternates ]
CLUE: <pyramidal tossup>
"""

SYSTEM_PROMPT_WRITER = """
You are an expert Quizbowl tossup writer. Given a TOPIC and 3 niche FACTS,
write a complete pyramidal tossup. Use the supplied facts as the lead-in
(hardest clues), then add 1-2 commonly-known facts of your own to lead into
the giveaway.

### Tossup Style (HS / NAQT level)
- 3-4 sentences, ~80-130 words.
- Pyramidal: supplied niche facts first, then your easier facts.
- Refer to the answer only as "this <category>".
- End with: "For 10 points, name this <category> <giveaway description>."
- Never include the answer, alias, abbreviation, eponym, synonym, or translation
  (e.g. "printf debugging" leaks "print debugging"). No "also known as".
  If an input fact would leak, paraphrase it.

### Answer line
List every common alternate name in square brackets, separated by "; or".
This is what the moderator will accept as a correct buzz. Examples:
- "Mark Twain [or Samuel Clemens; or Samuel Langhorne Clemens]"
- "Aliens trick [or WQS binary search; or Lagrange optimization on convex DP]"
- "Knuth–Morris–Pratt algorithm [or KMP]"
If there are genuinely no alternates, just write the name with no brackets.
If a "DO NOT pick" list is given and the TOPIC turns out to alias one of those
names, output exactly: ANSWER: SKIP (and no clue). Otherwise list alternates;
plain answers without alternates need no brackets.

### Example
INPUT:
TOPIC: Leo Tolstoy
FACTS:
- 1901 excommunication by Russian Orthodox Holy Synod after publishing "Resurrection"
- Novella with Praskovya Fedorovna tormented by a dying judge's three-day scream
- Konstantin Levin proposes to Kitty by writing first letters of words in chalk
- Novel opening "All happy families are alike"
- Russian author of War and Peace and Anna Karenina

OUTPUT:
ANSWER: Leo Tolstoy [or Lev Nikolayevich Tolstoy; or Lev Tolstoy]
CLUE: A 1901 excommunication by the Russian Orthodox Holy Synod followed this man's publication of a novel in which he satirized the church through the prostitute Maslova. In a novella by this author, Praskovya Fedorovna is tormented by the three-day scream of a dying judge. He created Konstantin Levin, who proposes to Kitty Shcherbatskaya by writing the first letters of words in chalk. He opened another novel by declaring that "all happy families are alike; each unhappy family is unhappy in its own way." For 10 points, name this Russian author of War and Peace and Anna Karenina.

### Output (no preamble)
ANSWER: <answer>[ optional bracket of alternates ]
CLUE: <tossup>
"""

SYSTEM_PROMPT_JUDGE = """
Quizbowl judge. Does USER_GUESS name the same specific thing as ANSWER?
ANSWER may list accepted alternates after "| accept:" — match any of them.

CORRECT if the guess differs only by:
- capitalization or punctuation  ("palindrome" = "Palindrome", "twicetagram" = "Twicetagram")
- notation/reformulation  ("p=np" = "P vs NP problem", "WWII" = "World War II")
- last name only  ("Tolstoy" = "Leo Tolstoy")
- common abbreviation or alias  ("KMP" = "Knuth-Morris-Pratt", "Samuel Clemens" = "Mark Twain")
- roman numeral / numeric variant  ("World War 2" = "World War II")

INCORRECT if the guess is:
- a different specific thing in the same field  ("merge sort" ≠ "quicksort", "Faraday's law" ≠ "Lenz's law")
- a broader category or parent concept  ("sorting" ≠ "quicksort")
- a sibling concept or related tool that is not the answer
- obviously a guess that is not related, a long shot or BS answer

Default INCORRECT when unsure. Output exactly:
RESULT: CORRECT or INCORRECT
FEEDBACK: One sentence.
"""

class Brain:
    def __init__(self):
        self.groq = groq_client
        self.gemini = gemini_client

    def _client_for(self, model: str):
        """Route to the right OpenAI-compatible endpoint by model name."""
        return self.gemini if model.startswith(("gemini", "gemma")) else self.groq

    async def _chat(self, model: str, messages: list, **kwargs):
        """Chat completion with automatic fallback for the fast Llama model
        on rate-limit / quota errors. Returns (response, model_actually_used).

        The OpenAI SDK is synchronous, so we offload to a thread — without
        this, a single API call blocks the entire asyncio event loop and
        nothing else (streaming, broadcasts, other handlers) can run.
        """
        try:
            resp = await asyncio.to_thread(
                self._client_for(model).chat.completions.create,
                model=model, messages=messages, **kwargs,
            )
            return resp, model
        except Exception as e:
            s = str(e).lower()
            rate_limited = any(k in s for k in ("429", "rate_limit", "quota", "tpd", "rpm"))
            if rate_limited and model == MODEL_WRITER_FAST:
                fb = MODEL_WRITER_SLOW
                print(f"[fallback] {model} → {fb} (rate-limited)")
                resp = await asyncio.to_thread(
                    self._client_for(fb).chat.completions.create,
                    model=fb, messages=messages, **kwargs,
                )
                return resp, fb
            raise

    @staticmethod
    def _avoid_block(avoid: Optional[List[str]]) -> str:
        if not avoid:
            return ""
        recent = list(dict.fromkeys(avoid))[-100:]
        return "\n\nDO NOT pick any of these (already used):\n" + "\n".join(f"- {t}" for t in recent)

    async def generate_topic_list(self, general_topic: str, avoid: Optional[List[str]] = None, n: int = 5):
        """Architect: returns a list of {name, facts} dicts."""
        user_msg = (
            f"Subject: {general_topic}. "
            f"Pick {n} distinct topics and provide fact sheets."
            f"{self._avoid_block(avoid)}"
        )
        response = await asyncio.to_thread(
            self._client_for(MODEL_ARCHITECT).chat.completions.create,
            model=MODEL_ARCHITECT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ARCHITECT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            # Gemini 3.x OpenAI-compat: maps to thinking_level. "medium" gives
            # the model some room to reason about topic diversity / avoid list
            # without the unbounded scratchpad that Gemma 4 26B was producing.
            reasoning_effort="medium",
        )
        text = _strip_thinking(response.choices[0].message.content or "")
        raw_arch = response.choices[0].message.content or ""
        print(f"\n{'='*60}\n[ARCHITECT – {MODEL_ARCHITECT}] (n={n}, avoid={len(avoid or [])})\n{raw_arch}\n{'='*60}\n")
        topics = []
        for part in text.split("TOPIC:")[1:]:
            lines = part.split("\n")
            name = lines[0].strip()
            facts = [line.strip("- ").strip() for line in lines[1:] if line.strip().startswith("-")]
            if name:
                topics.append({"name": name, "facts": "\n".join(facts)})
        return topics

    async def generate_topic_and_clue(self, general_topic: str, avoid: Optional[List[str]] = None,
                                       model: str = MODEL_WRITER_FAST):
        """Combined writer: pick topic + write tossup in one call. For bootstrap.
        Returns (topic_name, primary_answer, aliases, clue) or (None, None, None, None)."""
        user_msg = (
            f"Subject: {general_topic}. Pick one specific topic and write a tossup."
            f"{self._avoid_block(avoid)}"
        )
        response, used = await self._chat(
            model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_WRITER_PICK},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
        )
        text = _strip_thinking(response.choices[0].message.content or "")
        print(f"\n{'='*60}\n[WRITER-PICK – {used}] (avoid={len(avoid or [])})\n{text}\n{'='*60}\n")
        try:
            topic = text.split("TOPIC:", 1)[1].split("ANSWER:", 1)[0].strip().split("\n")[0].strip()
            answer_line = text.split("ANSWER:", 1)[1].split("CLUE:", 1)[0].strip().split("\n")[0].strip()
            clue = text.split("CLUE:", 1)[1].strip()
            primary, aliases = _parse_answer_line(answer_line)
            if topic and primary and clue:
                return topic, primary, aliases, clue
        except (IndexError, AttributeError):
            pass
        return None, None, None, None

    async def generate_clue_from_facts(self, topic_data: dict,
                                        avoid: Optional[List[str]] = None,
                                        model: str = MODEL_WRITER_SLOW):
        """Steady-state writer: turn architect's topic+facts into a tossup.
        Receives the avoid list too so it can flag aliasing collisions
        (output 'ANSWER: SKIP' if the topic aliases something already used).
        Returns (primary_answer, aliases, clue) or (None, None, None)."""
        user_msg = (
            f"TOPIC: {topic_data['name']}\nFACTS:\n{topic_data['facts']}"
            f"{self._avoid_block(avoid)}"
        )
        response, used = await self._chat(
            model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_WRITER},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
        )
        text = _strip_thinking(response.choices[0].message.content or "")
        print(f"\n{'='*60}\n[WRITER – {used}] topic='{topic_data.get('name')}'\n{text}\n{'='*60}\n")
        try:
            answer_line = text.split("ANSWER:", 1)[1].split("CLUE:", 1)[0].strip().split("\n")[0].strip()
            # Writer signals "this topic aliases something on the avoid list"
            if answer_line.upper().startswith("SKIP"):
                print(f"[writer] SKIP requested for '{topic_data.get('name')}' (alias collision)")
                return None, None, None
            clue = text.split("CLUE:", 1)[1].strip()
            primary, aliases = _parse_answer_line(answer_line)
            if primary and clue:
                return primary, aliases, clue
        except (IndexError, AttributeError):
            pass
        return None, None, None

    async def validate_answer(self, answer: str, aliases: List[str], guess: str, clue: str):
        # Stage 1: cheap token-set prefilter against primary AND every alias.
        # Catches "P = NP" / "P vs NP problem", "Lenz" / "Lenz's law", and
        # any writer-supplied alternate that normalizes the same.
        candidates = [answer] + list(aliases or [])
        for cand in candidates:
            if _quick_match(cand, guess):
                print(f"[JUDGE – prefilter] ACCEPT '{guess}' ≈ '{cand}' (token-equal)")
                return True, "Correct."

        # Stage 2: strict LLM judge — sees every accepted alias so it
        # doesn't have to guess. Format: "Primary | accept: alt1; alt2".
        if aliases:
            answer_block = f"{answer} | accept: " + "; ".join(aliases)
        else:
            answer_block = answer
        response = await asyncio.to_thread(
            self.groq.chat.completions.create,
            model=MODEL_JUDGE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                {"role": "user", "content": f"ANSWER: {answer_block} | USER_GUESS: {guess} | CLUE: {clue}"},
            ],
            temperature=0,
        )
        text = _strip_thinking(response.choices[0].message.content or "")
        print(f"\n{'='*60}\n[JUDGE – {MODEL_JUDGE}]\nANSWER: {answer_block} | GUESS: {guess}\n{text}\n{'='*60}\n")
        try:
            result = text.split("RESULT:")[1].split("FEEDBACK:")[0].strip()
            feedback = text.split("FEEDBACK:")[1].strip()
            return result.upper().startswith("CORRECT"), feedback
        except IndexError:
            return False, "Validation error."

# ── Gameplay constants (server-authoritative) ───────────────────────
BUZZ_WINDOW_MS = 10000   # post-reading buzz window
ANSWER_WINDOW_MS = 10000 # time the buzzer has to answer
WORD_PACE_S = 0.20

# ── Topic queue tuning ──────────────────────────────────────────────
BOOTSTRAP_TARGET     = 3   # ready clues to produce via fast combined writer
TOPUP_TOPIC_FETCH    = 3   # architect batch size in steady state
LOW_QUEUE_THRESHOLD  = 5   # top up when (topic_queue + ready clues) < this
FAST_MODEL_THRESHOLD = 3   # writer uses fast model when ready clues ≤ this

# ── Persistent global "already used" topic log ──────────────────────
USED_TOPICS_FILE = "used_topics.json"
USED_TOPICS_CAP = 30  # only the most recent N globally — keeps prompts small


def load_used_topics() -> List[str]:
    try:
        with open(USED_TOPICS_FILE, "r") as f:
            data = json.load(f)
            return list(data.get("topics", []))[-USED_TOPICS_CAP:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_used_topics(topics: List[str]) -> None:
    try:
        with open(USED_TOPICS_FILE, "w") as f:
            json.dump({"topics": topics[-USED_TOPICS_CAP:]}, f, indent=2)
    except Exception as e:
        print(f"[used_topics] save failed: {e}")

# ── Phases ──────────────────────────────────────────────────────────
# idle       : no game running
# loading    : round started, generating topics
# reading    : streaming words
# buzz_window: streaming finished, players can still buzz
# answering  : someone has buzzed, they must answer
# revealed   : question resolved (correct / dead)
PHASE_IDLE = 'idle'
PHASE_LOADING = 'loading'
PHASE_READING = 'reading'
PHASE_BUZZ_WINDOW = 'buzz_window'
PHASE_ANSWERING = 'answering'
PHASE_REVEALED = 'revealed'


class GameState:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_ids: Dict[WebSocket, str] = {}
        # connection_counts[name] = number of open sockets for that player
        self.connection_counts: Dict[str, int] = {}

        # Phase + answer state
        self.phase: str = PHASE_IDLE
        self.winner: Optional[str] = None
        self.current_answer: Optional[str] = None       # primary, displayed
        self.current_answer_aliases: List[str] = []     # accepted alternates
        self.current_clue: Optional[str] = None

        # Background tasks
        self.streaming_task: Optional[asyncio.Task] = None
        self.bg_generator_task: Optional[asyncio.Task] = None
        self.timer_task: Optional[asyncio.Task] = None

        # Server-side timer (for late-join sync)
        self.timer_kind: Optional[str] = None      # 'buzz_window' | 'answer'
        self.timer_started: Optional[float] = None # time.monotonic()
        self.timer_duration_s: float = 0.0

        # Scoring
        self.scores: Dict[str, int] = {}

        # Topic queue
        self.topic_queue: List[Dict] = []
        self.pregenerated_clues: asyncio.Queue = asyncio.Queue()

        # Per-question state
        self.buzzed_players: set = set()
        self.current_words: List[str] = []
        self.stream_position: int = 0
        self.question_number: int = 0

        # Topic deduplication
        self.current_round_topic: Optional[str] = None
        self.session_used_topics: List[str] = []   # this round's used names
        self.global_used_topics: List[str] = load_used_topics()  # cross-session, persisted

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_ids[websocket] = client_id

        # Track per-player connection counts; only fire join when this is the
        # first socket for the player (prevents 2nd-tab "joined" noise).
        prev = self.connection_counts.get(client_id, 0)
        self.connection_counts[client_id] = prev + 1
        unique_count = len(self.connection_counts)
        if prev == 0:
            await self.broadcast({"type": "player_joined", "player": client_id, "count": unique_count})

        # Always send a state_sync so the new client sees current players + scores,
        # and the partial clue if a game is in progress.
        sync: dict = {
            "type": "state_sync",
            "players": sorted(self.connection_counts.keys()),
            "scores": self.scores,
            "question_number": self.question_number,
            "phase": self.phase,
        }
        # Attach live clue + winner when a question is mid-flight
        if self.phase in (PHASE_READING, PHASE_BUZZ_WINDOW, PHASE_ANSWERING):
            sync["partial_clue"] = " ".join(self.current_words[:self.stream_position])
            sync["words_read"] = self.stream_position
            sync["winner"] = self.winner
        # Attach timer remaining if one is active
        remaining = self.timer_remaining_ms()
        if remaining > 0 and self.timer_kind:
            sync["timer"] = {"kind": self.timer_kind, "remaining_ms": remaining}
        await websocket.send_text(json.dumps(sync))

    async def disconnect(self, websocket: WebSocket):
        client_id = self.client_ids.pop(websocket, "unknown")
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Decrement; only fire player_left when the player has zero open tabs
        if client_id in self.connection_counts:
            self.connection_counts[client_id] -= 1
            if self.connection_counts[client_id] <= 0:
                del self.connection_counts[client_id]
                await self.broadcast({
                    "type": "player_left",
                    "player": client_id,
                    "count": len(self.connection_counts),
                })
                # Critical: if the disappearing player was the active buzzer,
                # force a wrong answer so the game doesn't soft-lock.
                if self.phase == PHASE_ANSWERING and self.winner == client_id:
                    asyncio.create_task(force_wrong_answer(client_id, ""))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

    async def broadcast_except(self, message: dict, exclude: WebSocket):
        for connection in self.active_connections:
            if connection is exclude:
                continue
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

    def reset_buzzer(self):
        self.winner = None

    async def stop_streaming(self):
        if self.streaming_task:
            self.streaming_task.cancel()
            self.streaming_task = None

    async def cancel_timer(self):
        if self.timer_task:
            self.timer_task.cancel()
            self.timer_task = None
        self.timer_kind = None
        self.timer_started = None
        self.timer_duration_s = 0.0

    def timer_remaining_ms(self) -> int:
        if not self.timer_kind or self.timer_started is None:
            return 0
        elapsed = time.monotonic() - self.timer_started
        return max(0, int((self.timer_duration_s - elapsed) * 1000))

game = GameState()
brain = Brain()
app = FastAPI()

if not API_SHARED_TOKEN:
    print("[security] API_SHARED_TOKEN is empty; WebSocket password gate is disabled")


def _is_token_valid(token: str) -> bool:
    # Empty server token means "auth disabled" for local development.
    if not API_SHARED_TOKEN:
        return True
    return bool(token) and token == API_SHARED_TOKEN


# ── Server-authoritative gameplay helpers ─────────────────────────
async def start_buzz_window():
    """Reading is done. Open a 10s buzz window; declare dead on expiry."""
    await game.cancel_timer()
    game.phase = PHASE_BUZZ_WINDOW
    game.timer_kind = "buzz_window"
    game.timer_started = time.monotonic()
    game.timer_duration_s = BUZZ_WINDOW_MS / 1000
    await game.broadcast({
        "type": "buzz_window",
        "duration_ms": BUZZ_WINDOW_MS,
    })
    game.timer_task = asyncio.create_task(_buzz_window_expire())


async def _buzz_window_expire():
    try:
        await asyncio.sleep(BUZZ_WINDOW_MS / 1000)
        if game.phase == PHASE_BUZZ_WINDOW:
            await declare_dead()
    except asyncio.CancelledError:
        pass


async def start_answer_timer(buzzer: str):
    """Buzzer has 10s to submit an answer. Force-wrong on expiry."""
    await game.cancel_timer()
    game.phase = PHASE_ANSWERING
    game.timer_kind = "answer"
    game.timer_started = time.monotonic()
    game.timer_duration_s = ANSWER_WINDOW_MS / 1000
    game.timer_task = asyncio.create_task(_answer_timer_expire(buzzer))


async def _answer_timer_expire(buzzer: str):
    try:
        await asyncio.sleep(ANSWER_WINDOW_MS / 1000)
        if game.phase == PHASE_ANSWERING and game.winner == buzzer:
            await force_wrong_answer(buzzer, "")
    except asyncio.CancelledError:
        pass


async def force_wrong_answer(buzzer: str, guess: str):
    """Resolve a buzz as wrong (user submitted, timed out, or disconnected).
    Handles score, broadcast, and the next phase transition."""
    if game.phase != PHASE_ANSWERING or game.winner != buzzer:
        return
    await game.cancel_timer()

    game.scores[buzzer] = game.scores.get(buzzer, 0) - 5
    game.buzzed_players.add(buzzer)

    await game.broadcast({
        "type": "wrong_buzz",
        "guesser": buzzer,
        "guess": guess,
        "scores": game.scores,
    })
    game.reset_buzzer()

    # Everyone wrong → end the question
    online = set(game.client_ids.values())
    if online and game.buzzed_players >= online:
        await declare_dead()
        return

    # More words to read → resume streaming; otherwise re-open buzz window
    if game.stream_position < len(game.current_words):
        game.phase = PHASE_READING
        game.streaming_task = asyncio.create_task(resume_streaming())
    else:
        await start_buzz_window()


async def declare_dead():
    """No correct answer obtained. Reveal the answer and freeze."""
    if game.phase == PHASE_REVEALED:
        return
    await game.cancel_timer()
    game.phase = PHASE_REVEALED
    await game.broadcast({
        "type": "question_dead",
        "answer": game.current_answer,
        "answer_aliases": game.current_answer_aliases,
        "full_clue": game.current_clue,
    })

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    token = websocket.query_params.get("token", "")
    if not _is_token_valid(token):
        # Policy violation
        await websocket.close(code=1008)
        return

    await game.connect(websocket, client_id)
    if client_id not in game.scores:
        game.scores[client_id] = 0

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            # Refresh in case a rename in another tab changed our identity
            client_id = game.client_ids.get(websocket, client_id)

            mtype = message.get("type")

            if mtype == "start":
                topic = message.get("topic", "General Knowledge")
                await game.stop_streaming()
                await game.cancel_timer()
                game.reset_buzzer()
                game.phase = PHASE_LOADING
                game.question_number = 0
                game.current_words = []
                game.stream_position = 0
                game.buzzed_players = set()
                game.session_used_topics = []  # fresh session avoid-list
                game.current_round_topic = topic

                game.topic_queue = []
                while not game.pregenerated_clues.empty():
                    game.pregenerated_clues.get_nowait()

                await game.broadcast({"type": "round_started", "topic": topic, "starter": client_id})

                # bg_generator owns the architect call (initial fetch + top-up)
                if game.bg_generator_task:
                    game.bg_generator_task.cancel()
                game.bg_generator_task = asyncio.create_task(background_generator(topic))
                game.streaming_task = asyncio.create_task(stream_next_clue(client_id))

            elif mtype == "next":
                await game.stop_streaming()
                await game.cancel_timer()
                game.reset_buzzer()
                game.buzzed_players = set()
                game.phase = PHASE_LOADING
                game.streaming_task = asyncio.create_task(stream_next_clue(client_id))

            elif mtype == "buzz":
                if game.phase not in (PHASE_READING, PHASE_BUZZ_WINDOW):
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "buzz_phase",
                        "message": "Can't buzz right now."
                    }))
                elif client_id in game.buzzed_players:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "buzz_again",
                        "message": "Already buzzed this question."
                    }))
                else:
                    await game.stop_streaming()
                    await game.cancel_timer()
                    game.winner = client_id
                    await game.broadcast({
                        "type": "lock",
                        "winner": client_id,
                        "answer_window_ms": ANSWER_WINDOW_MS,
                    })
                    await start_answer_timer(client_id)

            elif mtype == "submit_answer":
                if game.phase != PHASE_ANSWERING or game.winner != client_id:
                    continue  # stale submit; ignore

                guess = message.get("guess", "").strip()
                if not guess:
                    is_correct, feedback = False, "No answer given."
                else:
                    is_correct, feedback = await brain.validate_answer(
                        game.current_answer, game.current_answer_aliases,
                        guess, game.current_clue,
                    )

                if is_correct:
                    await game.cancel_timer()
                    game.scores[client_id] = game.scores.get(client_id, 0) + 10
                    game.buzzed_players.add(client_id)
                    game.phase = PHASE_REVEALED
                    await game.broadcast({
                        "type": "reveal",
                        "correct": True,
                        "answer": game.current_answer,
                        "answer_aliases": game.current_answer_aliases,
                        "feedback": feedback,
                        "winner": client_id,
                        "guess": guess,
                        "full_clue": game.current_clue,
                        "scores": game.scores,
                    })
                    game.reset_buzzer()
                else:
                    # All wrong-answer paths funnel through force_wrong_answer
                    # (handles score, broadcast, transition to next phase).
                    await force_wrong_answer(client_id, guess)

            elif mtype == "reset":
                await game.stop_streaming()
                await game.cancel_timer()
                if game.bg_generator_task:
                    game.bg_generator_task.cancel()
                    game.bg_generator_task = None
                game.reset_buzzer()
                game.buzzed_players = set()
                game.scores = {}
                game.question_number = 0
                game.phase = PHASE_IDLE
                game.current_words = []
                game.stream_position = 0
                game.session_used_topics = []
                game.current_round_topic = None
                game.topic_queue = []
                while not game.pregenerated_clues.empty():
                    game.pregenerated_clues.get_nowait()
                await game.broadcast({"type": "reset"})

            elif mtype == "rename":
                new_name = (message.get("name") or "").strip()[:24]
                if not new_name or new_name == client_id:
                    continue
                # Conflict only if a *different* player already holds that name
                others = {n for ws_, n in game.client_ids.items() if ws_ is not websocket}
                if new_name in others:
                    await websocket.send_text(json.dumps({
                        "type": "error", "code": "rename_conflict",
                        "message": "Name already taken",
                    }))
                    continue

                old_name = client_id
                # Update ALL sockets of this player (multi-tab support)
                for ws_key in list(game.client_ids.keys()):
                    if game.client_ids[ws_key] == old_name:
                        game.client_ids[ws_key] = new_name
                # Migrate state keyed by name
                if old_name in game.scores:
                    game.scores[new_name] = game.scores.pop(old_name)
                else:
                    game.scores.setdefault(new_name, 0)
                if old_name in game.buzzed_players:
                    game.buzzed_players.discard(old_name)
                    game.buzzed_players.add(new_name)
                if game.winner == old_name:
                    game.winner = new_name
                if old_name in game.connection_counts:
                    game.connection_counts[new_name] = game.connection_counts.pop(old_name)
                client_id = new_name

                await game.broadcast({
                    "type": "player_renamed",
                    "old": old_name,
                    "new": new_name,
                    "scores": game.scores,
                })

    except WebSocketDisconnect:
        await game.disconnect(websocket)

def _track_session(name: str) -> None:
    """Mark a topic as used in this session (so the architect / writer-pick
    won't pick it again this round). Called as soon as a topic is selected,
    even before it's been streamed."""
    if name and name not in game.session_used_topics:
        game.session_used_topics.append(name)


def _commit_global(name: str) -> None:
    """Persist a topic to the cross-session global log. Called only when the
    user actually starts seeing the clue — so unused/cancelled topics don't
    waste slots in the avoid list."""
    if not name:
        return
    # Move-to-front: if it was already in there, refresh its position
    if name in game.global_used_topics:
        game.global_used_topics.remove(name)
    game.global_used_topics.append(name)
    if len(game.global_used_topics) > USED_TOPICS_CAP:
        game.global_used_topics = game.global_used_topics[-USED_TOPICS_CAP:]
    save_used_topics(game.global_used_topics)


def _avoid_list() -> List[str]:
    """Combined session + global topics to avoid (deduped, ordered)."""
    return list(dict.fromkeys(game.session_used_topics + game.global_used_topics))


async def background_generator(topic: str):
    """Two-phase generator for the question pipeline.

    Phase 1 (bootstrap): until BOOTSTRAP_TARGET clues are ready, the writer
    picks a topic AND writes the tossup in a single fast call (Llama 70B).
    This minimizes time-to-first-clue.

    Phase 2 (steady state): architect picks 5 topics in one call (Gemma 27B);
    writer turns each topic+facts into a tossup. Writer model is depth-aware:
    fast Llama when ready clues ≤ FAST_MODEL_THRESHOLD, slow Gemini Flash Lite
    when above (saves Groq quota).
    """
    bootstrap_done = False
    try:
        while True:
            ready = game.pregenerated_clues.qsize()

            # ── Phase 1: bootstrap with combined fast writer ──
            if not bootstrap_done:
                if ready >= BOOTSTRAP_TARGET:
                    bootstrap_done = True
                    continue
                try:
                    name, answer, aliases, clue = await brain.generate_topic_and_clue(
                        topic, avoid=_avoid_list(), model=MODEL_WRITER_FAST,
                    )
                except Exception as e:
                    print(f"[bootstrap] {e}")
                    await asyncio.sleep(2)
                    continue
                # Dedup against session topics by primary AND any alias
                # (catches "Lev Tolstoy" coming back when "Leo Tolstoy" used)
                duplicate = name in game.session_used_topics or any(
                    a in game.session_used_topics for a in (aliases or [])
                )
                if name and answer and clue and not duplicate:
                    game.pregenerated_clues.put_nowait((name, answer, aliases, clue))
                    _track_session(name)
                    print(f"[bootstrap] +1 '{name}' aliases={aliases} (ready={game.pregenerated_clues.qsize()})")
                else:
                    reason = "duplicate alias" if duplicate else "no usable result"
                    print(f"[bootstrap] {reason}, retrying")
                    await asyncio.sleep(1)
                continue

            # ── Phase 2: architect → writer pipeline ──
            pool = ready + len(game.topic_queue)

            # Top up topics from the architect: only when total pool is low
            # AND the queue has been drained (no point asking for more raw
            # topics while writer is still working through a previous batch).
            if pool < LOW_QUEUE_THRESHOLD and not game.topic_queue:
                try:
                    topics = await brain.generate_topic_list(
                        topic, avoid=_avoid_list(), n=TOPUP_TOPIC_FETCH,
                    )
                except Exception as e:
                    print(f"[architect] {e}")
                    await asyncio.sleep(3)
                    continue
                added = 0
                for t in topics:
                    name = (t.get("name") or "").strip()
                    if name and name not in game.session_used_topics:
                        game.topic_queue.append(t)
                        _track_session(name)
                        added += 1
                print(f"[architect] queued {added}/{len(topics)} topics")

            # Process queued topics into clues whenever there's something to
            # process — independent of pool depth. Pool only gates the
            # architect; the writer should always be filling the pipeline.
            # Writer model is depth-aware: fast (Llama) when ready ≤ 3, slow
            # (Gemini Flash Lite) when there's enough buffer to absorb it.
            if game.topic_queue:
                td = game.topic_queue.pop(0)
                use_fast = ready <= FAST_MODEL_THRESHOLD
                writer_model = MODEL_WRITER_FAST if use_fast else MODEL_WRITER_SLOW
                try:
                    answer, aliases, clue = await brain.generate_clue_from_facts(
                        td, avoid=_avoid_list(), model=writer_model,
                    )
                except Exception as e:
                    print(f"[writer] '{td.get('name')}': {e}")
                    await asyncio.sleep(2)
                    continue
                if not (answer and clue):
                    continue
                # Dedup: if writer revealed an alias matching a used topic,
                # drop this clue rather than queue a duplicate.
                name = td.get("name") or answer
                used = set(game.session_used_topics) - {name}  # don't reject self
                if name in used or any(a in used for a in (aliases or [])):
                    print(f"[writer] dropping '{name}' — alias collides with prior topic")
                    continue
                game.pregenerated_clues.put_nowait((name, answer, aliases, clue))
                print(f"[writer] +1 '{name}' aliases={aliases} via {writer_model} "
                      f"(ready={game.pregenerated_clues.qsize()})")
            else:
                await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass

async def stream_next_clue(client_id: str):
    print(f"[stream] waiting for clue (queue={game.pregenerated_clues.qsize()}, conns={len(game.active_connections)})")
    try:
        try:
            name, answer, aliases, clue = await asyncio.wait_for(
                game.pregenerated_clues.get(), timeout=30
            )
        except asyncio.TimeoutError:
            print(f"[stream] TIMEOUT waiting for clue")
            game.phase = PHASE_IDLE
            await game.broadcast({"type": "error", "message": "AI is taking too long. Try a different topic."})
            return

        print(f"[stream] got clue: '{answer}' aliases={aliases} ({len(clue.split())} words)")
        # User is about to see this clue — commit to the persistent avoid list
        _commit_global(name)
        game.current_answer = answer
        game.current_answer_aliases = aliases or []
        game.current_clue = clue
        game.current_words = clue.split()
        game.stream_position = 0
        game.buzzed_players = set()
        game.question_number += 1
        game.phase = PHASE_READING
        print(f"[stream] broadcasting question_start (Q{game.question_number}, conns={len(game.active_connections)})")
        await game.broadcast({"type": "question_start", "question_number": game.question_number})

        for i, word in enumerate(game.current_words):
            if game.phase != PHASE_READING:  # buzzed mid-stream
                game.stream_position = i
                return
            await game.broadcast({"type": "clue_chunk", "text": word})
            game.stream_position = i + 1
            await asyncio.sleep(WORD_PACE_S)

        print(f"[stream] reading complete, opening buzz window")
        if game.phase == PHASE_READING:
            await start_buzz_window()

    except asyncio.CancelledError:
        print(f"[stream] cancelled")
        raise
    except Exception as e:
        import traceback
        print(f"[stream] ERROR: {e}\n{traceback.format_exc()}")
        await game.broadcast({"type": "error", "message": "AI failed to stream clue."})


async def resume_streaming():
    """Resume word-by-word reading after a wrong buzz."""
    try:
        for i in range(game.stream_position, len(game.current_words)):
            if game.phase != PHASE_READING:
                game.stream_position = i
                return
            await game.broadcast({"type": "clue_chunk", "text": game.current_words[i]})
            game.stream_position = i + 1
            await asyncio.sleep(WORD_PACE_S)

        if game.phase == PHASE_READING:
            await start_buzz_window()
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
