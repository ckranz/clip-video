# `re-burn-captions` Command - Design Document

## Problem

After refining transcripts with the `refine` command, existing highlight clips still have captions burned from the original (pre-refinement) transcript text. The clip extraction and portrait conversion are fine (based on timestamps from LLM analysis), but the captions need re-burning with the corrected text.

## Solution

Add a `clip-video re-burn-captions BRAND` command that reloads refined transcripts and re-burns captions on existing highlight clips without re-extracting or re-analyzing.

## Command Signature

```
clip-video re-burn-captions BRAND [--project TEXT]
```

## Behavior

1. Load brand config
2. Find all highlight project directories under `brands/{brand}/highlights/` (or filter by `--project` substring match)
3. For each project:
   - Load `HighlightsProject` from `project_state.json`
   - Load the refined transcript from `brands/{brand}/transcripts/{video_stem}.json`
   - Back up existing final clips:
     - First reburn: copy to `clips/backups/pre-reburn/`
     - Subsequent: copy to `clips/backups/reburn-YYYY-MM-DDTHHMMSS/`
   - Clear `captioned_clip_path` on each clip so `burn_captions` doesn't skip them
   - Call `processor.burn_captions(project, transcript_segments)`
   - Regenerate social copy if enabled (also uses transcript text)
   - Save project state
4. Print per-project summary and overall summary

## What Changes, What Doesn't

- **Raw clips** (`clips/raw/`): untouched
- **Portrait clips** (`clips/portrait/`): untouched
- **Final clips** (`clips/final/`): regenerated with updated captions
- **Social copy** (`metadata/`): regenerated with updated transcript text
- **Backups** (`clips/backups/`): archived copies of previous final clips

## Transcript Matching

The project's `video_path` maps to a transcript file. Extract the video stem from `project.video_path` and look for `brands/{brand}/transcripts/{stem}.json`.

## Backup Strategy

- First reburn: copy existing finals to `clips/backups/pre-reburn/`
- Subsequent reburns: copy to `clips/backups/reburn-YYYY-MM-DDTHHMMSS/`

Mirrors the transcript refinement backup approach.

## Error Handling

- Skip projects with no clips (not yet processed)
- Skip projects where transcript file can't be found (warn and continue)
- Continue on individual project failure
