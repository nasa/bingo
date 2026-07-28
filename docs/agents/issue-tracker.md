# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files under `docs/issues/`.

## Conventions

- One initiative per directory: `docs/issues/<initiative-slug>/`
- The PRD is `docs/issues/<initiative-slug>/PRD.md`
- Implementation issues are `docs/issues/<initiative-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Each issue file starts with YAML frontmatter containing at least `title`, `category`, `state`, `blocked_by`, and `created`
- The body stays human-readable markdown below the frontmatter
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `docs/issues/<initiative-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path directly, or identify the initiative and issue number.
