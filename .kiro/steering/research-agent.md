---
inclusion: always
---

# Research agent baseline

This workspace has a project-level agent brief at `#[[file:../../AGENTS.md]]`. Read it for
repository conventions, and follow the `deep-research` skill at
`#[[file:../../.agents/skills/deep-research/SKILL.md]]` in full whenever a task involves
research, literature review, source-backed investigation, or fact-finding of any kind.

Key non-negotiables from that skill, restated here so they apply even to quick requests:

- Never fabricate a citation, URL, author, date, or statistic. Mark unverifiable claims
  `[unverified]` instead.
- After any download, verify the file actually contains what it claims (check size and
  real file type) before recording it as successfully downloaded. See
  `#[[file:../../.agents/skills/deep-research/references/download-verification-checklist.md]]`.
- Maintain a running source log as you research, not reconstructed from memory afterward.
  Template: `#[[file:../../.agents/skills/deep-research/references/source-log-template.md]]`.
- Do a self-review pass before presenting findings: trace claims back to logged sources,
  confirm scope was fully covered, confirm deliverable files exist and are valid.
- If an approach fails twice (a search, a download, a command), diagnose the root cause
  before trying a third time rather than repeating the same failing approach.

## Committing work

Follow `#[[file:../../.agents/skills/git-commit/SKILL.md]]` for when and how to commit.
In short: commit at real checkpoints (a task/sub-task completes and is verified, before
a risky change, before a handoff or interruption, or on explicit request) — not on a
timer and not only at the very end of a session. Stage specific files by name, never
`git add .`/`-A`, scan for secrets before staging, write a message that explains the
change and why, and verify the commit afterward with `git log -1 --stat`. Never push,
force-push, or rewrite history without explicit instruction.

## Keeping this infrastructure current (strict)

`AGENTS.md`, `.agents/skills/`, and this steering file are living documents, not a
one-time setup. Every session that touches them must follow `AGENTS.md`'s "Maintaining
this file and the skills library" section in full:

- Install a new skill under `.agents/skills/<name>/SKILL.md` when a task reveals a
  genuinely reusable procedure — not for one-off work.
- Fix an existing skill immediately, in the same session, if using it surfaces a gap —
  don't leave it for later.
- Log every change to `AGENTS.md`, any `SKILL.md`, or any `references/` file in
  `#[[file:../../.agents/CHANGELOG.md]]` before considering the task done.
- Before ending a session that touched `.agents/`, re-check that the skills table in
  `AGENTS.md` and this steering file both still match what's actually on disk.
