# gutenberg-reader

Convert any [Project Gutenberg](https://www.gutenberg.org/) book into structured JSON ready for text-to-speech (TTS) audiobook generation with [tts-audiobook](https://github.com/anders94/tts-audiobook). Each sentence is labelled as **narration** or **dialogue**, with speaker attribution and pronunciation hints, so TTS engines can apply per-character voices automatically.

Processing runs entirely locally against any OpenAI-compatible LLM server ([vLLM](https://docs.vllm.ai/), llama.cpp server, LM Studio, ...) — no cloud API keys required.

---

## Features

- Downloads and processes any Gutenberg book by numeric ID
- **Deterministic segmentation**: narration/dialogue splitting is done by quote parsing, not an LLM — text coverage is guaranteed and reported speech ("Mr. Bennet replied that he had not.") can never be misclassified as dialogue
- Layered speaker attribution: deterministic attribution-tag anchors, LLM resolution of nameless tags ("said his lady"), deterministic alternation propagation, then LLM review of only the unresolved turns
- **Guided decoding**: speaker names are constrained to the known character list via vLLM structured output — invented or misspelled speakers are impossible
- Identifies characters and their aliases across the full book
- Resumable pipeline: interrupted runs pick up where they left off
- Atomic cache writes — safe to kill at any point

---

## Requirements

- Python 3.11+
- An OpenAI-compatible LLM server. For example, vLLM:

```bash
vllm serve google/gemma-4-E4B-it-qat-w4a16-ct \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template vllm-env/tool_chat_template_gemma4.jinja
```

Any server exposing `/v1/models` and `/v1/chat/completions` with `response_format` JSON-schema support works (vLLM, llama.cpp server, LM Studio).

---

## Installation

```bash
pip install gutenberg-reader
```

Or install from source:

```bash
git clone https://github.com/your-username/gutenberg-reader
cd gutenberg-reader
pip install -e .
```

---

## Quick start

```bash
# Process Pride and Prejudice (book 1342) — model auto-detected from the server
gutenberg-reader 1342

# Process Dr. Jekyll and Mr. Hyde with verbose output
gutenberg-reader 43 --verbose

# Use a specific model / server
gutenberg-reader 1342 --base-url http://localhost:8000/v1 --model google/gemma-4-E4B-it-qat-w4a16-ct
```

Output is written to `cache/<book_id>/07-final/<book_id>.json`.

---

## CLI reference

```
gutenberg-reader BOOK_ID [OPTIONS]
```

`BOOK_ID` is the numeric Project Gutenberg book identifier. Find it in the book's URL:
`https://www.gutenberg.org/ebooks/1342` → ID is `1342`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url URL` | `http://localhost:8000/v1` | OpenAI-compatible API base URL |
| `--api-key KEY` | `EMPTY` | API key, if the server requires one |
| `--model MODEL` | *(auto-detect)* | Model for attribution and character discovery; defaults to the first model the server offers |
| `--validator MODEL` | *(same as --model)* | Separate model for the Stage 06 critic pass |
| `--cache-dir DIR` | `./cache` | Directory for all cached stage outputs |
| `--output FILE` | *(auto)* | Override output JSON path |
| `--chunk-size N` | `1000` | Words per LLM attribution window |
| `--critic` | off | Run the Stage 06 LLM critic pass (most useful with a larger `--validator` model) |
| `--force-stage N` | — | Re-run from stage N (1–7) forward, discarding cached results from that stage on |
| `--chapters N[,N,…]` | — | Process only specific chapter numbers (e.g. `1,2,5`) |
| `--max-retries N` | `3` | Max LLM retries per attribution window |
| `-v / --verbose` | off | Show per-stage progress and correction details |

### Examples

```bash
# Add the LLM critic pass, reviewing with a larger model on another machine
gutenberg-reader 1342 --critic --validator some-larger-model --verbose

# Re-run only the assembly stage (Stage 07) to regenerate final JSON
gutenberg-reader 1342 --force-stage 7

# Re-run segmentation + everything after, for chapters 1–3 only
gutenberg-reader 1342 --force-stage 5 --chapters 1,2,3

# Point at a bigger model on another machine (e.g. a Mac Studio running llama.cpp/LM Studio)
gutenberg-reader 1342 --base-url http://192.168.1.10:8080/v1

# Write output to a custom path
gutenberg-reader 1342 --output ~/audiobooks/pride_and_prejudice.json
```

---

## Model recommendations

Because segmentation is deterministic and the LLM only resolves speaker
attribution (with guided decoding constraining outputs), model requirements are
much lighter than for full LLM segmentation. On a single 24 GB GPU:

| Model | Quality | Notes |
|-------|---------|-------|
| `google/gemma-4-E4B-it-qat-w4a16-ct` | ★★★★☆ | Fast; fine for most books |
| Gemma 3 27B (W4A16) | ★★★★★ | Best attribution on ambiguous multi-party scenes |
| Qwen 2.5 14B (AWQ/GPTQ) | ★★★★☆ | Good balance |

For maximum quality on books with large ensemble casts, point `--validator` (or
the whole run) at a larger model served from a bigger machine — any
OpenAI-compatible endpoint works.

---

## Output format

The final JSON has this structure:

```json
{
  "metadata": {
    "title": "The Strange Case of Dr. Jekyll and Mr. Hyde",
    "author": "Robert Louis Stevenson",
    "language": "English",
    "gutenberg_id": "43",
    "release_date": "June 27, 2008",
    "credits": "David Widger"
  },
  "chapters": [
    {
      "chapter": {
        "number": 1,
        "title": "STORY OF THE DOOR",
        "start_line": 42,
        "end_line": 310,
        "word_count": 2150
      },
      "processed": {
        "chapter_number": 1,
        "chapter_title": "STORY OF THE DOOR",
        "segments": [
          {
            "type": "narration",
            "text": "Mr. Utterson the lawyer was a man of a rugged countenance that was never lighted by a smile;",
            "speaker": null,
            "pronunciation_hints": [],
            "notes": null
          },
          {
            "type": "dialogue",
            "text": "“I incline to Cain’s heresy,”",
            "speaker": "Mr. Utterson",
            "pronunciation_hints": [],
            "notes": null
          },
          {
            "type": "narration",
            "text": "he used to say quaintly:",
            "speaker": null,
            "pronunciation_hints": [],
            "notes": null
          },
          {
            "type": "dialogue",
            "text": "“I let my brother go to the devil in his own way.”",
            "speaker": "Mr. Utterson",
            "pronunciation_hints": [],
            "notes": null
          }
        ],
        "word_count": 2150
      },
      "validation": null
    }
  ],
  "characters": [
    {
      "name": "Mr. Utterson",
      "aliases": [],
      "pronunciation_hints": [],
      "first_appearance_chapter": 1
    },
    {
      "name": "Dr. Jekyll",
      "aliases": ["Henry Jekyll"],
      "pronunciation_hints": [],
      "first_appearance_chapter": 1
    }
  ],
  "statistics": {
    "total_chapters": 10,
    "total_words": 25615,
    "total_segments": 1315,
    "total_characters": 8,
    "processing_time_seconds": 777.3,
    "validation_performed": true,
    "pipeline_version": "1.0.0"
  }
}
```

### Segment types

| `type` | `speaker` | Description |
|--------|-----------|-------------|
| `"narration"` | `null` | All non-spoken text: description, action, attribution tags |
| `"dialogue"` | `"Character Name"` | Text inside quotation marks, spoken aloud by a character |

Attribution phrases like *"said Mr. Bennet"* or *"replied his wife"* are always `"narration"` segments, never merged into the surrounding dialogue. This lets TTS engines handle attribution in a natural voice separate from the speaking characters.

When the speaker cannot be determined, `speaker` is `"Unknown"`.

A dialogue segment whose quotation continues into the next paragraph (Gutenberg
convention: no closing quote at paragraph end) carries `"notes": "quote-continues"`;
the following dialogue segment is the same speaker.

---

## Pipeline stages

The pipeline runs in 7 stages. Each stage writes to `cache/<book_id>/0N-<name>/` and is skipped on subsequent runs if its output already exists.

| Stage | Name | Description |
|-------|------|-------------|
| 01 | Download | Fetches `pg<id>.txt` from Gutenberg with retry/backoff |
| 02 | Discovery | Strips boilerplate, detects chapter boundaries, extracts metadata |
| 03 | Chapter split | Extracts each chapter into a plain-text file |
| 04 | Characters | LLM identifies all named characters and aliases |
| 05 | Segmentation | Deterministic quote-based narration/dialogue split + three-tier speaker attribution |
| 06 | Critic | Deterministic anchor-propagation pass + optional LLM attribution review |
| 07 | Assembly | Merges all chapters into the final JSON |

### Resume and force-rerun

Every stage checks for its cached output before running. If you interrupt the pipeline (Ctrl+C), restarting the same command resumes from the interrupted chapter automatically.

To reprocess from a specific stage:

```bash
# Regenerate final JSON only
gutenberg-reader 1342 --force-stage 7

# Re-run critic pass and reassemble
gutenberg-reader 1342 --force-stage 6

# Start completely from scratch
gutenberg-reader 1342 --force-stage 1
```

### Cache layout

```
cache/
└── 1342/
    ├── 01-raw/          book.txt
    ├── 02-discovery/    discovery.json
    ├── 03-chapters/     chapter-01.txt … chapter-61.txt
    ├── 04-characters/   characters.json
    ├── 05-segments/     chapter-01.json … chapter-61.json
    ├── 06-critic/       chapter-01.json … chapter-61.json
    └── 07-final/        1342.json
```

---

## How attribution works

Narration/dialogue **segmentation is not an LLM task**: quotation marks delimit
dialogue, so the split is done by a deterministic parser (`segmenter.py`) that
handles curly and straight quotes, apostrophes, and multi-paragraph quotations.
Every character of the source text lands in exactly one segment, verbatim.

Speaker attribution then runs in four tiers:

1. **Attribution anchors (deterministic):** narration like *"said Mr. Bennet"*
   adjacent to a dialogue segment confirms its speaker. The speech verb must
   lead a sentence of the tag (so scene beats containing a verb mid-sentence
   don't false-match), reported speech (*"replied that ..."*) is excluded, and
   a tag ending in a period only anchors backwards. Names after a speech verb
   anchor even when character discovery missed them (*"said Lydia,"*).

2. **Tag resolution (LLM, constrained):** nameless tags (*"said his lady"*,
   *"returned she"*) are resolved to character names — a far easier task than
   free attribution — and then anchor their adjacent dialogue.

3. **Anchor propagation (deterministic):** consecutive dialogue is grouped into
   *conversation chains*; confirmed speakers propagate via strict alternation in
   2-character scenes. Characters present in a scene are detected from vocative
   name use inside dialogue (e.g., *"My dear Mr. Bennet,"* confirms Mr. Bennet
   is present — as the listener).

4. **LLM review:** only dialogue still unresolved is sent to the LLM, in
   windows with surrounding context. Guided decoding constrains each answer to
   the known character list (or `"Unknown"`), so the model cannot invent names.

Stage 06 optionally runs an LLM critic over each chapter's final attribution;
it may only *reassign speakers* (again constrained to the character list) —
segment text is never touched by any model. Disable with `--no-critic`.

---

## Tests and fixtures

```bash
pytest                                  # fixtures that are missing are skipped
GUTENBERG_FIXTURES=required pytest      # missing fixtures fail instead (use in CI)
python scripts/fetch_fixtures.py        # restore the books the tests are built on
python scripts/build_fixtures.py        # regenerate goldens + candidate lists
```

The regression suite is built on real books, but `cache/` is gitignored, so the
books themselves are not in the repo — only what is *derived* from them:

- `tests/fixtures/golden/<id>.json` — where the chapters are, per book.
- `tests/fixtures/candidates/<id>.txt` — the condensed view a structure pass is
  shown. Committed so a change to candidate extraction shows up as a reviewable
  diff rather than a silent shift in what the model sees.
- `tests/fixtures/manifest.json` — a SHA-256 per book. A re-download that differs
  is reported instead of being used, since a re-issued text moves every boundary
  in its golden.

Fetch the books with `scripts/fetch_fixtures.py` before relying on a green run.
Without them the fixture-backed tests skip by default, which is quiet enough to
miss — `GUTENBERG_FIXTURES=required` turns that into a failure.

---

## Limitations

- Requires a locally running OpenAI-compatible LLM server
- Books that mark dialogue without quotation marks (em-dash dialogue, some
  translations) fall back to narration-only segmentation
- Books in languages other than English work if the model supports the language, but the speech-verb detection regex is English-only
- Some books use non-standard chapter structures (Roman numerals, titled chapters, etc.) — the pipeline falls back to LLM-based chapter discovery when regex detection finds fewer than 2 chapters

---

## License

MIT — see [LICENSE](LICENSE).
