# gutenberg-reader

Convert any [Project Gutenberg](https://www.gutenberg.org/) book into structured JSON ready for text-to-speech (TTS) audiobook generation with [tts-audiobook](https://github.com/anders94/tts-audiobook). Each sentence is labelled as **narration** or **dialogue**, with speaker attribution and pronunciation hints, so TTS engines can apply per-character voices automatically.

Processing runs entirely locally using [Ollama](https://ollama.com/) — no cloud API keys required.

---

## Features

- Downloads and processes any Gutenberg book by numeric ID
- Splits text into narration and dialogue segments with speaker attribution
- Identifies characters and their aliases across the full book
- Deterministic anchor-propagation pass corrects misattributed back-and-forth dialogue before the LLM review
- Resumable pipeline: interrupted runs pick up where they left off
- Atomic cache writes — safe to kill at any point

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)
- A capable instruction-following model — see [Model recommendations](#model-recommendations)

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
# Process Pride and Prejudice (book 1342)
gutenberg-reader 1342

# Process Dr. Jekyll and Mr. Hyde with verbose output
gutenberg-reader 43 --verbose

# Use a specific model
gutenberg-reader 1342 --model gemma3:27b --verbose
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
| `--model MODEL` | `qwen2.5:14b` | Ollama model for segmentation and character discovery |
| `--validator MODEL` | *(same as --model)* | Separate model for the Stage 06 critic pass |
| `--ollama-url URL` | `http://localhost:11434` | Ollama API base URL |
| `--cache-dir DIR` | `./cache` | Directory for all cached stage outputs |
| `--output FILE` | *(auto)* | Override output JSON path |
| `--chunk-size N` | `400` | Words per processing chunk (lower = more API calls, less context pressure) |
| `--overlap N` | `150` | Overlap words between chunks |
| `--no-critic` | off | Skip Stage 06 LLM critic pass (faster; deterministic anchor pass still runs) |
| `--force-stage N` | — | Re-run from stage N (1–7) forward, discarding cached results from that stage on |
| `--chapters N[,N,…]` | — | Process only specific chapter numbers (e.g. `1,2,5`) |
| `--max-retries N` | `3` | Max LLM retries per chunk on integrity failure |
| `-v / --verbose` | off | Show per-stage progress and correction details |

### Examples

```bash
# Fast run: skip LLM critic, use deterministic anchor pass only
gutenberg-reader 1342 --no-critic --verbose

# Use a larger model for segmentation, lighter model for critic
gutenberg-reader 1342 --model gemma3:27b --validator gemma3:12b

# Re-run only the assembly stage (Stage 07) to regenerate final JSON
gutenberg-reader 1342 --force-stage 7

# Re-run segmentation + everything after, for chapters 1–3 only
gutenberg-reader 1342 --force-stage 5 --chapters 1,2,3

# Point at a remote Ollama instance
gutenberg-reader 1342 --ollama-url http://192.168.1.10:11434 --model llama3.3:70b

# Write output to a custom path
gutenberg-reader 1342 --output ~/audiobooks/pride_and_prejudice.json
```

---

## Model recommendations

Any Ollama model that follows structured JSON instructions will work. Tested configurations:

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| `gemma3:27b` | ★★★★★ | Slow | Best attribution accuracy |
| `qwen2.5:14b` | ★★★★☆ | Medium | Good default choice |
| `gemma3:12b` | ★★★★☆ | Fast | Good balance |
| `gemma4:latest` | ★★★★☆ | Fast | Tested reference model |
| `llama3.3:70b` | ★★★★★ | Very slow | For maximum quality |

Pull a model before first use:

```bash
ollama pull qwen2.5:14b
```

Smaller models (7B and below) tend to produce more JSON formatting errors and speaker misattributions. The pipeline has automatic repair and retry logic, but larger models need fewer corrections.

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
            "text": "\u201cI incline to Cain\u2019s heresy,\u201d",
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
            "text": "\u201cI let my brother go to the devil in his own way.\u201d",
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

---

## Pipeline stages

The pipeline runs in 7 stages. Each stage writes to `cache/<book_id>/0N-<name>/` and is skipped on subsequent runs if its output already exists.

| Stage | Name | Description |
|-------|------|-------------|
| 01 | Download | Fetches `pg<id>.txt` from Gutenberg with retry/backoff |
| 02 | Discovery | Strips boilerplate, detects chapter boundaries, extracts metadata |
| 03 | Chapter split | Extracts each chapter into a plain-text file |
| 04 | Characters | LLM identifies all named characters and aliases |
| 05 | Segmentation | LLM splits each chapter into narration/dialogue segments with speaker attribution |
| 06 | Critic | Deterministic anchor-propagation pass + optional LLM quality review |
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

Back-and-forth dialogue that lacks explicit attribution tags (e.g., *"said Mr. Bennet"*) is handled in two passes:

1. **Anchor propagation (Stage 06, deterministic):** The pipeline locates every narration segment containing a speech verb adjacent to dialogue. It groups consecutive dialogue into *conversation chains*, then uses confirmed speakers as anchors and propagates via strict alternation for 2-character scenes. Characters present in a scene are detected from vocative name use inside dialogue (e.g., *"My dear Mr. Bennet,"* confirms Mr. Bennet is present).

2. **LLM critic (Stage 06, optional):** A second LLM pass reviews the anchor-corrected segments for remaining attribution issues, name inconsistencies, and coverage gaps. Disable with `--no-critic` for faster processing.

---

## Limitations

- Requires a locally running Ollama instance
- Processing a full novel takes 15–90 minutes depending on model and hardware
- Very long chapters (>5,000 words) are chunked; chunk boundaries can occasionally split a dialogue exchange across two LLM calls
- Books in languages other than English work if the model supports the language, but the speech-verb detection regex is English-only
- Some books use non-standard chapter structures (Roman numerals, titled chapters, etc.) — the pipeline falls back to LLM-based chapter discovery when regex detection finds fewer than 2 chapters

---

## License

MIT — see [LICENSE](LICENSE).
