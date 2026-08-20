# Agent Infrastructure Changelog

Every creation or edit to `AGENTS.md`, anything under `.agents/`, or the
`.kiro/steering/research-agent.md` pointer must get a dated entry here in the same
session as the change, per the maintenance rules in `AGENTS.md`. This file is the audit
trail for the agent infrastructure itself — treat gaps in it the same way you'd treat
gaps in a research source log: as a fumble to avoid, not a formality to skip.

## Entry format

```
## YYYY-MM-DD — <one-line summary>

- Changed: <exact file(s) touched, with paths>
- Why: <what prompted the change — a task, a gap found, a new recurring need>
- Details: <what actually changed, specific enough that someone reading only this
  entry understands the change without opening a diff>
```

New entries go at the top (most recent first).

---

## 2026-08-20 — Added git-commit skill

- Changed:
  - `.agents/skills/git-commit/SKILL.md` (created)
  - `.agents/skills/git-commit/references/checkpoint-triggers-checklist.md` (created)
  - `.agents/skills/git-commit/references/commit-message-examples.md` (created)
  - `AGENTS.md` (edited — added `git-commit` row to skills table, added a "commit at
    real checkpoints" bullet under general agent behavior)
  - `.kiro/steering/research-agent.md` (edited — added "Committing work" section
    pointing at the new skill)
  - `.agents/CHANGELOG.md` (this entry)
- Why: User asked for a git-and-commit skill plus instructions so commits happen
  properly and at the right times going forward, rather than work sitting uncommitted
  or getting committed carelessly.
- Details: Wrote a `git-commit` skill covering: confirming repo/branch state before
  acting (Phase 0); recognizing real commit checkpoints vs. arbitrary timing (Phase 1 —
  task/sub-task done and verified, before a risky change, before a handoff, or on
  explicit request; not mid-edit and not on a timer); staging deliberately by named
  paths (never `git add .`/`-A`) with an explicit secret-scan step (Phase 2); message
  format and content rules (Phase 3); post-commit verification via `git log -1 --stat`
  instead of trusting the command's exit status alone (Phase 4); and stricter rules for
  pushing/remotes (never force-push, never push to main without being asked, never skip
  hooks) (Phase 5). Registered it in `AGENTS.md`'s skills table and referenced it from
  the always-on steering file so it actually gets applied automatically, following the
  same wiring pattern used for `deep-research`.

## 2026-08-20 — Initial agent infrastructure created

- Changed:
  - `AGENTS.md` (created)
  - `.agents/skills/deep-research/SKILL.md` (created)
  - `.agents/skills/deep-research/references/source-log-template.md` (created)
  - `.agents/skills/deep-research/references/source-credibility-rubric.md` (created)
  - `.agents/skills/deep-research/references/download-verification-checklist.md` (created)
  - `.kiro/steering/research-agent.md` (created)
- Why: User asked for a proper research skill to be installed and an `AGENTS.md` set up
  so the agent would follow a rigorous, non-fumbling research procedure going forward,
  based on the literature-survey work already completed in `literature survey/`.
- Details: Researched the open Agent Skills format (agentskills.io / anthropics/skills)
  and several community deep-research SKILL.md implementations for structural reference,
  then wrote an original `deep-research` skill tailored to this environment's actual
  tools (no Claude-Code-specific subagent/slash-command dependencies). The skill defines
  six phases (scope, search, fan-out, verify, log, self-review, deliver) plus explicit
  anti-patterns. `AGENTS.md` was written following the open agents.md convention,
  pointing to the skill and stating repo-wide non-negotiables (no fabricated citations,
  verify downloads, log sources as you go, self-review before presenting, diagnose
  repeated failures instead of retrying blindly). A Kiro steering file
  (`.kiro/steering/research-agent.md`, `inclusion: always`) was added because Kiro does
  not natively auto-load `AGENTS.md` or `.agents/skills/` — without it those files would
  be inert and never actually read by the agent in this tool.

## 2026-08-20 — Added strict maintenance rules and this changelog

- Changed:
  - `AGENTS.md` (edited — added "Maintaining this file and the skills library" section,
    added a pointer to this changelog under "Where to look first")
  - `.agents/CHANGELOG.md` (created, this file)
- Why: User asked for strict instructions on keeping skills/instructions (including
  `.agents/` and `AGENTS.md` itself) updated as work continues, with a maintained log of
  changes, and for new skills to be installed wherever a task reveals the need for one.
- Details: Added explicit rules to `AGENTS.md`: install a new skill when a task reveals a
  genuinely reusable procedure (but not for one-off work); fix an existing skill the
  moment a gap is found while using it, in the same session; keep one concern per skill
  folder; log every change to agent infrastructure here before considering a task done;
  keep the skills table and the steering pointer in sync with what's actually on disk;
  keep "Project conventions" accurate as the repo evolves. Created this changelog with a
  standard entry format and backfilled the entry for the initial setup so the history is
  complete from the start rather than starting mid-story.
