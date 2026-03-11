# Standalone `refine` Command - Design Document

## Problem

The LLM transcript refinement feature only runs as part of the `transcribe` command, requiring a full Whisper re-run to refine existing transcripts. Users with already-transcribed brands (e.g., 33 KCD-UK talks) need to apply LLM refinement without re-transcribing.

## Solution

Add a standalone `clip-video refine BRAND` command that loads existing transcript JSON files and runs only the LLM refinement step.

## Command Signature

```
clip-video refine BRAND [--video TEXT] [--provider TEXT] [--model TEXT]
                        [--talk-title TEXT] [--talk-description TEXT]
```

## Behavior

1. Load brand config and vocabulary
2. Find all transcript JSON files in `brands/{brand}/transcripts/` (or filter to `--video` substring match)
3. For each transcript:
   - Load via `TranscriptionResult.load()`
   - **Backup**: If no `.pre-refine.json` exists, save one (original preservation). If `.pre-refine.json` already exists, save a timestamped backup (`.refine-YYYY-MM-DDTHHMMSS.json`)
   - Run `TranscriptRefiner.refine()` with context
   - If corrections found: update segments, text, and `vocabulary_corrections` count; save back to original path
   - Print summary per file (corrections count or "no changes")
4. Print overall summary (files processed, total corrections)

## Provider Selection

Same as existing `--refine` on transcribe: CLI flags override brand config's `llm_provider`/`llm_model`.

## `--video` Filter

Substring match against transcript filename (without extension). So `--video "David Flanagan"` matches `David Flanagan - Extreme Microservices...json`.

## Backup Strategy

- **First run**: saves `{name}.pre-refine.json` — the original pre-LLM version, preserved permanently
- **Subsequent runs**: saves `{name}.refine-YYYY-MM-DDTHHMMSS.json` — timestamped snapshot before each re-refinement

This gives: original Whisper output + history of each refinement pass.

## Error Handling

- If LLM fails on one transcript, log warning and continue to next
- Never raises on individual transcript failure
- Print skip message for transcripts with no corrections

## Files Changed

- `src/clip_video/cli.py` — new `refine` command
- No new modules needed — reuses `TranscriptRefiner`, `RefinementContext`, `TranscriptionResult`
