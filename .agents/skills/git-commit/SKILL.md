---
name: git-commit
description: |
  Rules and a phased procedure for when and how to make git commits during a task:
  choosing the right commit boundaries, writing clear commit messages, staging safely,
  and never committing secrets or unrelated changes. Use this skill whenever meaningful
  file changes have been made and a natural checkpoint has been reached (a task or
  sub-task completes, a logical unit of work is done, before starting a risky or
  unrelated change), or whenever the user asks to commit, save progress, or create a
  checkpoint. Also covers repository initialization when a workspace has no git repo yet.
---

# Git Commit

This skill defines when a commit should happen, not just how to write `git commit`. The
failure mode this guards against isn't bad git syntax — it's an agent that either never
commits (losing checkpoints, making history unreadable when it finally does commit one
giant blob) or commits recklessly (secrets, unrelated changes bundled together, commits
on `main` that should've been on a branch). Follow this every time file changes are made,
not only when the user explicitly says "commit."

## Phase 0 — Confirm there is a repository, and that it's the right one

1. Check whether the current workspace is already a git repository (`git status`). If it
   is not, and the workspace clearly represents one coherent project (not a scratch/temp
   directory), initializing one is reasonable — but say so explicitly rather than doing it
   silently, since it's the kind of structural change a user should know happened.
2. If a repository already exists, confirm you're looking at the right one and the right
   branch before doing anything — check `git status` and `git branch --show-current`
   every time you're about to commit, don't assume the state hasn't changed since the
   last check (other tools, other sessions, or the user may have touched it).
3. Never modify git config, and never use destructive history-rewriting commands
   (`reset --hard`, `push --force`, `clean -fd`, `branch -D`, `commit --amend` on commits
   that aren't your own unpushed work) without the user explicitly asking for that specific
   operation. Committing is additive and safe; almost everything else in git is not.

## Phase 1 — Recognize when "the time is right" to commit

Commit at natural checkpoints, not on an arbitrary schedule and not only at the very end
of a session. A checkpoint is reached when any of these are true:

- A discrete task or clearly-scoped sub-task has been completed and verified (code
  builds/tests pass, a document is finished, a feature works end-to-end).
- You are about to start a materially different or riskier change, and want a clean
  rollback point before doing so.
- A meaningful, self-contained unit of work exists that would be painful to lose or hard
  to review if bundled with whatever comes next (e.g. "set up the agent skills
  infrastructure" is its own checkpoint, separate from "add the literature survey").
- The user explicitly asks to commit, save, or checkpoint progress.
- A long-running or multi-step task is about to be interrupted or handed off (session
  ending, context compaction approaching, delegating to a sub-agent).

Do **not** commit:

- Mid-edit, with a file in a known-broken/inconsistent state, unless explicitly told to
  commit a WIP snapshot (and if so, say `WIP:` in the message so it's clear later).
- Just because time has passed — a commit boundary is defined by logical completeness,
  not a timer.
- As a substitute for finishing verification. Verify first (per the project's own
  verification norms), then commit the verified result.

## Phase 2 — Stage deliberately

1. **Never use `git add .` or `git add -A` by default.** Stage specific files/paths by
   name. This is the single biggest guard against accidentally committing unrelated
   in-progress changes, scratch files, or secrets that happen to be sitting in the
   working tree.
2. Before staging, run `git status` and actually read the list — don't stage blind.
3. **Scan for secrets before staging anything new.** Flag and exclude: `.env` files,
   credential/token files, private keys, anything matching typical secret patterns. If a
   file legitimately contains a secret and must exist in the repo, it should be in
   `.gitignore`, not staged — raise this to the user if you're not sure.
4. If unrelated changes exist in the working tree alongside the change you're
   checkpointing (e.g. an unrelated file was touched by an earlier experiment), stage only
   what belongs to this checkpoint. Split into multiple commits rather than one commit
   mixing unrelated concerns.
5. Check for a `.gitignore`; if one doesn't exist and the project has obvious
   noise-to-exclude (build artifacts, dependency directories, OS files, editor files),
   create or extend one before the first commit rather than after noise is already tracked.

## Phase 3 — Write the message

Format:

```
<short summary, imperative mood, ~50-70 chars>

<body: what changed and why, wrapped at ~72 chars, if the change isn't
self-evident from the summary alone>
```

Rules:

- Summary line describes the *change*, not the mechanism ("Add literature survey on AV
  object detection" not "Ran fs_write a bunch of times").
- If the commit bundles more than one file but they're all part of one logical change,
  the body can briefly enumerate the parts — but if it needs a bulleted list to explain
  unrelated things, that's a signal it should have been split into separate commits
  (Phase 2, point 4) instead.
- Do not editorialize about how the work was done or restate the diff; explain intent
  and effect.
- Do not include AI/assistant attribution boilerplate unless the project's own
  conventions ask for it.

## Phase 4 — Commit, then verify

1. Run the commit.
2. Immediately run `git log -1 --stat` (or `git show --stat HEAD`) and check it actually
   contains what you intended — right files, right message, nothing extra swept in. A
   commit command returning success is not sufficient proof it committed the right
   content; verify the same way any other file operation gets verified.
3. Do **not** push automatically. Pushing is a separate, explicit decision — see Phase 5.

## Phase 5 — Pushing and remotes (higher risk, more caution)

- Never push directly to `main`/`master` unless the user explicitly asks for that. Default
  to a new branch and `git push -u origin <branch>` for tracking.
- Never force-push, and never skip hooks (`--no-verify`) unless explicitly asked.
- If a pre-commit/pre-push hook fails, fix the underlying issue, re-stage, and make a new
  commit — do not `--amend` past a hook failure unless told to.
- Only create commits when it's clear the user wants a checkpoint saved — but note that
  routine, expected checkpoints (per Phase 1) don't require asking permission each time if
  the user has already established that this workflow should run; asking every single
  time defeats the point of "commit when the time is right." Pushing to a remote is a
  bigger action and warrants more explicit confirmation than a local commit does.

## Anti-patterns

- Never committing anything across an entire multi-hour task, then presenting one giant
  commit with an uninformative message at the very end.
- Using `git add .` reflexively instead of naming files.
- Committing a broken/half-finished state without a `WIP:` marker.
- Bundling unrelated changes (e.g. an agent-infrastructure update and an unrelated code
  fix) into one commit because both happened to be sitting in the working tree.
- Amending commits that have already been pushed, or amending after a hook failure.
- Committing secrets because they happened to be in a file that got swept up by a
  wildcard `git add`.
- Force-pushing or rewriting history "to clean things up" without being asked.

## Reference files

- `references/commit-message-examples.md` — good vs. bad commit message examples.
- `references/checkpoint-triggers-checklist.md` — quick checklist for Phase 1 decisions.
