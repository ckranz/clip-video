# Clip Video

Intelligent video clip extraction tool for lyric matching and highlights generation.

## Installation

```bash
pip install -e .
```

### Requirements

- **Python 3.10+**
- **FFmpeg** - Required for video processing (auto-downloaded on first run, or install manually)
- **API Keys** (set as environment variables):
  - `OPENAI_API_KEY` - For Whisper transcription
  - `ANTHROPIC_API_KEY` - For Claude-based highlight detection (optional, can use OpenAI instead)

## Quick Start

### 1. Initialize a Brand

A "brand" is a collection of videos from a single source (e.g., a conference, YouTube channel, podcast series):

```bash
clip-video init-brand "TechConf2024" --description "Tech Conference 2024 recordings"
```

This creates the folder structure:
```
~/.clip-video/brands/TechConf2024/
├── videos/          # Put your source videos here
├── transcripts/     # Auto-generated transcripts
├── projects/        # Lyric match projects
├── outputs/         # Generated clips
└── config.json      # Brand configuration
```

### 2. Add Source Videos

Copy or move your video files into the brand's `videos/` folder:

```bash
# Find your brand folder
clip-video info TechConf2024

# Copy videos into it
cp /path/to/videos/*.mp4 ~/.clip-video/brands/TechConf2024/videos/
```

Supported formats: MP4, MOV, AVI, MKV, WEBM

### 3. Transcribe Videos

Transcribe all videos in the brand (uses OpenAI Whisper API):

```bash
# Interactive mode - shows cost estimate and asks for confirmation
clip-video transcribe TechConf2024

# Skip confirmation (for batch/unattended runs)
clip-video transcribe TechConf2024 --yes

# Transcribe a specific video only
clip-video transcribe TechConf2024 --video "keynote.mp4"

# Force re-transcription
clip-video transcribe TechConf2024 --force
```

**Cost**: ~$0.006/minute of audio (Whisper API pricing)

---

## Highlights Mode

Generate social media shorts from conference recordings with burned-in captions.

### Single Video

```bash
# Generate 5 highlight clips (default)
clip-video highlights TechConf2024 "keynote.mp4"

# Generate 10 highlights
clip-video highlights TechConf2024 "keynote.mp4" --count 10

# With session description for better context
clip-video highlights TechConf2024 "keynote.mp4" --description session_info.txt
```

Output clips are saved to:
```
~/.clip-video/brands/TechConf2024/outputs/highlights/keynote/
├── highlight_001.mp4    # Portrait (9:16) with captions
├── highlight_002.mp4
├── ...
└── metadata.json        # Clip info, timestamps, titles
```

### Batch Processing (Weekend Runs)

Process multiple videos unattended:

```bash
# Create a list of videos to process
cat > videos_to_process.txt << EOF
session1_morning_keynote.mp4
session2_workshop_intro.mp4
session3_panel_discussion.mp4
day2_closing.mp4
EOF

# Run batch processing (resumes if interrupted)
clip-video highlights-batch TechConf2024 videos_to_process.txt --count 5
```

**Resume capability**: If the process is interrupted, just run the same command again - it will skip completed videos.

**Estimated cost per video** (30-minute recording):
- Transcription: ~$0.18 (if not already transcribed)
- LLM analysis: ~$0.05-0.15
- Total: ~$0.20-0.35 per video

---

## Lyric Match Mode

Build searchable word/phrase dictionaries for creating music video mashups.

### Search for Words

Find all occurrences of a word across your video library:

```bash
# Search for a word
clip-video search TechConf2024 "innovation"

# Limit results
clip-video search TechConf2024 "cloud" --limit 20

# Export matching clips
clip-video search TechConf2024 "future" --export
```

### Lyric Match Projects

Match lyrics to video clips:

```bash
# Create lyrics file
cat > song_lyrics.txt << EOF
We are the champions
My friends
And we'll keep on fighting
Till the end
EOF

# Start lyric match project
clip-video lyric-match TechConf2024 "champions_video" song_lyrics.txt

# Resume existing project
clip-video lyric-match TechConf2024 "champions_video" song_lyrics.txt --resume
```

---

## Configuration

### Brand Configuration

Edit `~/.clip-video/brands/<brand>/config.json`:

```json
{
  "name": "TechConf2024",
  "description": "Tech Conference 2024",
  "vocabulary": ["Kubernetes", "GraphQL", "WebAssembly"],
  "default_transcription_provider": "whisper_api",
  "default_llm_provider": "claude"
}
```

The `vocabulary` list helps correct domain-specific terms in transcriptions.

### API Keys

Create a `.env` file with your API keys. The tool checks these locations (in order):

1. `.env` in the current directory
2. `~/.clip-video/.env` (user-level config)

```bash
# Copy the example file
cp .env.example .env

# Edit with your keys
notepad .env  # or your editor
```

**`.env` contents:**
```bash
# Required for transcription (Whisper API)
OPENAI_API_KEY=sk-your-openai-key-here

# Required for Claude-based analysis (recommended)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

Or set as environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Workflow Examples

### Weekend Batch Run

Process an entire conference over the weekend:

```bash
# 1. Set up the brand
clip-video init-brand "PyCon2024"

# 2. Copy all videos
cp /path/to/pycon/*.mp4 ~/.clip-video/brands/PyCon2024/videos/

# 3. Transcribe everything first (can run overnight)
clip-video transcribe PyCon2024 --yes

# 4. Generate highlights (run over weekend)
ls ~/.clip-video/brands/PyCon2024/videos/*.mp4 | xargs -I{} basename {} > all_videos.txt
clip-video highlights-batch PyCon2024 all_videos.txt --count 5
```

### Check Progress

```bash
# See brand info and stats
clip-video info TechConf2024

# List all brands
clip-video list-brands-cmd
```

### Check Dependencies

```bash
clip-video check-deps
```

---

## Output Formats

### Highlight Clips

- **Format**: MP4 (H.264 + AAC)
- **Aspect Ratio**: 9:16 (portrait) - optimized for YouTube Shorts, TikTok, Instagram Reels
- **Captions**: Burned-in, white text with black outline
- **Duration**: 15-60 seconds (configurable)

### Metadata

Each highlight batch generates `metadata.json`:

```json
{
  "source_video": "keynote.mp4",
  "generated_at": "2024-01-15T10:30:00Z",
  "clips": [
    {
      "filename": "highlight_001.mp4",
      "title": "The Future of AI in Development",
      "start_time": 1234.5,
      "end_time": 1289.2,
      "duration": 54.7,
      "transcript": "..."
    }
  ]
}
```

---

## Tips

1. **Transcribe first**: Run transcription separately before highlights - it's reusable and the main cost.

2. **Session descriptions**: For better highlight detection, provide session descriptions:
   ```bash
   echo "Keynote about AI ethics and responsible development by Dr. Smith" > session_info.txt
   clip-video highlights TechConf2024 "keynote.mp4" --description session_info.txt
   ```

3. **Brand vocabulary**: Add domain terms to avoid transcription errors:
   ```json
   {
     "vocabulary": ["PyTorch", "TensorFlow", "Kubernetes", "gRPC"]
   }
   ```

4. **Check costs first**: The transcribe command shows cost estimates before proceeding.

5. **Resume-safe**: Both `transcribe` and `highlights-batch` can be safely interrupted and resumed.

---

## Troubleshooting

### FFmpeg not found

```bash
clip-video check-deps
# If FFmpeg is missing, it will be auto-downloaded on first video operation
```

### API errors

- Check your API keys are set correctly
- Rate limits: The tool automatically retries with backoff
- Cost limits: Monitor usage in your OpenAI/Anthropic dashboards

### Transcription issues

- Large files: Split into smaller segments if >2 hours
- Audio quality: Ensure clear audio for best results
- Force re-transcription: `clip-video transcribe BRAND --force`
