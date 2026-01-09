# .knowledge Folder

Portable knowledge base for Claude Code sessions. Copy this folder to any project to give Claude access to curated expertise.

## How It Works

1. Copy `.knowledge/` to your project root
2. Claude Code will read these files when relevant to your work
3. Reference specific files in CLAUDE.md if needed: `See .knowledge/writing.md for voice guidelines`

## Contents

| File | Purpose |
|------|---------|
| `writing.md` | Voice, tone, British English, AI tells to avoid |
| `seo.md` | SEO specifications and thresholds |
| `security.md` | Web security patterns and checklists |

## Adding Project-Specific Knowledge

Create additional `.md` files for project-specific knowledge:
- `brand.md` - Brand voice and messaging
- `architecture.md` - Project architecture decisions
- `api.md` - API conventions and patterns

## Usage in CLAUDE.md

```markdown
## Knowledge Base

This project uses a `.knowledge/` folder for curated expertise:
- Check `.knowledge/writing.md` before writing any content
- Check `.knowledge/security.md` when implementing auth or handling user input
- Check `.knowledge/seo.md` when working on pages or meta tags
```
