# LLM Fuzzy Word Matching for Lyric Match Mode

## Problem

When searching for lyric words across a brand's video library, some words have no exact match in any transcript. Currently the only options are adding more source videos, modifying lyrics, or manually adding vocabulary alternatives. For casual/slang words common in lyrics (gonna, ain't, wanna), the transcripts likely contain the formal equivalents (going to, isn't, want to) but there's no connection between them.

## Solution

Use the existing LLM infrastructure (Ollama/Claude/OpenAI) to opportunistically generate 1-3 fuzzy alternatives for missing words after initial search. Alternatives include phonetic matches (gonna -> connor), contractions/expansions (gonna -> going to), and casual speech equivalents (ain't -> isn't).

Enabled by default. Disable with `--no-fuzzy` flag.

## Architecture

### New module: `src/clip_video/lyrics/fuzzy.py`

`FuzzyWordExpander` class:
- Takes a list of missing words
- Sends them to the LLM in a single batch call
- Parses JSON response into `dict[str, list[str]]` mapping word -> alternatives
- Non-fatal: LLM failure logs a warning, returns empty dict

### Shared LLM caller: `src/clip_video/llm/caller.py`

Extract provider-dispatch logic from `TranscriptRefiner._call_llm` into a reusable `LLMCaller` class with a single method: `call(system_prompt, user_prompt) -> str`. `TranscriptRefiner._call_llm` delegates to it. Existing tests that mock `_call_llm` continue to pass unchanged.

### Integration point: `LyricMatchProcessor.search_all()`

After the initial search pass:
1. Identify missing single-word targets (no results, no alternatives with results)
2. If fuzzy enabled and missing words exist, call `FuzzyWordExpander`
3. Add returned alternatives to each `ExtractionTarget.alternatives`
4. Re-search just those targets using the new alternatives
5. Save project (alternatives persist in `project.json` via existing serialization)

On subsequent runs, words that already have alternatives skip the LLM call. The cached alternatives are searched via the existing alternatives code path in `search_all()`.

### CLI changes

- `LyricMatchConfig.fuzzy_matching: bool = True` (default enabled)
- `--no-fuzzy` flag on the `lyric-match` CLI command
- Console output: "Generating fuzzy alternatives for N missing words..." when triggered

### LLM prompt

System prompt instructs the LLM to suggest 1-3 alternatives per word: phonetic matches, contractions/expansions, casual speech equivalents. User prompt is the list of missing words. Response is JSON mapping each word to a list of alternatives.

### Error handling

LLM failure is non-fatal. If the call fails, log a warning and continue with exact matches only. Same resilience pattern as `TranscriptRefiner.refine()`.

## Data flow

```
Lyrics -> Parse -> Extract targets -> Initial search
                                          |
                                    Missing words?
                                     /         \
                                   No           Yes (and fuzzy enabled)
                                   |              |
                                   |         LLM: generate alternatives
                                   |              |
                                   |         Add to ExtractionTarget.alternatives
                                   |              |
                                   |         Re-search missing words
                                    \         /
                                     Results
```

## Testing

- `FuzzyWordExpander`: unit tests with mocked LLM (valid JSON, malformed, empty, error)
- `LLMCaller` extraction: verify existing `TranscriptRefiner` tests pass unchanged
- `search_all` integration: fuzzy runs when enabled, skips when disabled, skips when no missing words
- Verify alternatives persist in project.json and are not re-generated on subsequent runs
