# AGENTS.md

This file gives AI coding/research agents operating context for this repository. It
follows the open [AGENTS.md](https://agents.md) convention: a plain-Markdown, tool-agnostic
file that any compatible agent (Kiro, Claude Code, Cursor, Copilot, Codex, etc.) can read
for project-specific instructions.

## What this repository is

A workspace for research and analysis work. The current contents include a literature
survey on object detection for autonomous vehicles (`literature survey/`). Future work in
this repo may include other research or engineering tasks — treat this file as the
standing baseline, and anything in a task's own instructions as an override for that task.

## Skills

This repo bundles reusable agent skills under `.agents/skills/`, following the open
[Agent Skills](https://agentskills.io) format (a folder containing a `SKILL.md` with
metadata + instructions, plus optional `references/`, `scripts/`, `assets/`).

| Skill | Location | Use when |
|---|---|---|
| `deep-research` | `.agents/skills/deep-research/SKILL.md` | Any task involving research, literature review, source-backed investigation, or fact-finding that should produce a tracked, verifiable source trail rather than an off-the-cuff answer. |
| `git-commit` | `.agents/skills/git-commit/SKILL.md` | Whenever a meaningful checkpoint is reached (task/sub-task done and verified, before a risky change, before a handoff/interruption, or on explicit request) — decides when to commit, how to stage safely, and how to write the message. |

**Before starting any research task, read the relevant `SKILL.md` in full first.** It
defines a phased procedure (scope → search → verify → log → self-review → deliver)
specifically designed to prevent the failure modes that make research output untrustworthy:
fabricated sources, silently-broken downloads, unverifiable claims, and quietly-narrowed
scope. Do not shortcut it because a request "seems simple" — the phases that get skipped
under time pressure are exactly the ones that catch mistakes.

## Maintaining this file and the skills library (strict — do not skip)

`AGENTS.md`, everything under `.agents/`, and the `.kiro/steering/research-agent.md`
pointer are living infrastructure, not a one-time setup. They must be kept accurate and
current as work continues. This is not optional housekeeping — an agent brief that has
drifted from reality is worse than no brief at all, because it gives false confidence.

1. **Install a new skill when a task reveals a reusable procedure.** If you find yourself
   working out a non-trivial, repeatable procedure (a research method, a verification
   checklist, a recurring workflow), formalize it as a new skill folder under
   `.agents/skills/<skill-name>/SKILL.md`, following the same structure as `deep-research`
   (frontmatter `name` + `description`, phased instructions, optional `references/`).
   Add a row for it to the skills table above in the same session. Do not leave a
   newly-useful procedure undocumented just because the immediate task is finished.
2. **Do not formalize one-off work into a skill.** Only create a skill when the procedure
   is genuinely likely to recur. A single unusual task stays a one-off; do not bloat the
   skills library with narrow, non-reusable procedures.
3. **Fix a skill the moment it's found lacking.** If following an existing skill during
   real work surfaces a gap, ambiguity, wrong assumption, or missing edge case, edit that
   skill's `SKILL.md` (or its `references/`) immediately, in the same session — not a
   mental note, not a one-off workaround in the current task only. The next agent to load
   that skill should not hit the same gap.
4. **Keep skills scoped.** One concern per skill folder. If new instructions don't belong
   to an existing skill's concern, create a sibling skill rather than growing an existing
   one into an unrelated grab-bag.
5. **Log every change to this infrastructure.** Any creation or edit of a file under
   `.agents/` (including this `AGENTS.md` file itself, any `SKILL.md`, or any
   `references/` file) must get a dated entry in `.agents/CHANGELOG.md` before the task
   is considered done — what changed, in which file, and why. No silent edits to shared
   agent infrastructure. See `.agents/CHANGELOG.md` for the exact entry format.
6. **Keep the skills table and steering pointer in sync with disk.** Before ending any
   session that touched `.agents/`, re-list `.agents/skills/` and confirm the table above
   still lists every skill that exists, with an accurate one-line description of when to
   use it, and confirm `.kiro/steering/research-agent.md` still accurately reflects what's
   available. Stale pointers are a fumble in themselves — fix them in the same session
   they go stale, not "later."
7. **Update the "Project conventions" section as the project actually changes**, e.g. once
   real build/lint/test tooling exists, or new deliverable-folder conventions emerge. This
   file should always describe the repo as it actually is, not as it was when first written.

## General agent behavior for this repo

- **Never fabricate a citation, URL, statistic, author name, or file content.** If you
  cannot verify something, say so explicitly (`[unverified]`) instead of smoothing it over.
- **Verify file operations, especially downloads.** A tool call returning successfully is
  not proof a download or write actually produced valid content — check file size and
  actual content/type, not just that a path now exists. See
  `.agents/skills/deep-research/references/download-verification-checklist.md`.
- **Log sources as you go, not from memory afterward.** Any research task should leave
  behind a source log (link, what was done with it, a paraphrased summary, and the
  takeaway) alongside the findings themselves.
- **Do a self-review pass before presenting findings.** Re-trace each major claim back to
  a logged source; check you covered the scope you committed to; check deliverable files
  actually exist and aren't corrupt.
- **If an approach fails twice, diagnose before retrying a third time.** Don't repeat an
  identical failing shell command, download method, or search query expecting a different
  result — figure out *why* it failed and change the approach.
- **Respect licensing.** Only reuse images/datasets/text with a clear, checked license,
  and record attribution requirements when you do.
- **Prefer primary sources over secondary summaries**, and prefer multiple independent
  sources over one source repeated by aggregators.
- **Commit at real checkpoints, not arbitrarily.** Follow `.agents/skills/git-commit/SKILL.md`
  whenever a task or sub-task completes and is verified — don't let a whole session's
  work sit uncommitted, and don't commit broken/half-finished state either. See that
  skill for exactly when and how.

## Environment (read before running shell commands)

**The shell is fish, not bash.** This has bitten real sessions in this repo, so treat it as
a hard constraint rather than a footnote:

- **No bash-only syntax.** `declare -A` (associative arrays), `VAR=value` assignment,
  `export VAR=x`, `$?`, `${arr[@]}`, and bash heredocs all fail or behave differently in
  fish. Fish equivalents: `set VAR value`, `set -x VAR x`, `$status` instead of `$?`,
  `for x in ...; ...; end` instead of `done`.
- **For anything beyond a simple one-liner, wrap it explicitly:** `bash -c '<script>'`.
  This is the reliable escape hatch for loops, arrays, and multi-step scripts — use it
  rather than trying to hand-translate a script into fish.
- Simple chaining with `&&`, `||`, and `;` does work in fish 3.x, so short sequential
  commands are fine unwrapped.

**Other verified environment quirks in this workspace:**

- **The file-reading tools cannot read `/tmp`.** Redirecting command output to `/tmp/...`
  and then reading it back does not work here. Write scratch output inside the workspace
  instead (e.g. `./state.tmp`), read it, then delete it — and don't leave scratch files
  behind or commit them.
- **A freshly written file may not be readable on the very first attempt.** If a read
  immediately after a write reports the path as missing, retry once before concluding the
  write failed.
- **Terminal echo of commands renders garbled** in this setup (the command line is
  visually mangled in tool output). Judge success by the actual command output and exit
  code, not by the echoed command text looking wrong.

## Project conventions

- No build/test tooling exists yet in this repo (it currently holds research documents,
  not code). If code is added later, this section should be updated with the actual
  build/lint/test commands so agents don't have to rediscover them each session.
- Research deliverables live in their own top-level folder (e.g. `literature survey/`),
  each containing a source log and a findings document at minimum, plus `papers/` and
  `images/` subfolders for downloaded artifacts where relevant.

## Where to look first

- `.agents/skills/deep-research/SKILL.md` — the research procedure itself.
- `.agents/CHANGELOG.md` — dated history of every change to this file and the skills
  library, and why. Check it to understand how the agent infrastructure got to its
  current state, and add to it per the maintenance rules above.
- `literature survey/sources.md` — an example of a completed source log following this
  skill's conventions.
- `literature survey/findings.md` — an example of a completed findings document.
