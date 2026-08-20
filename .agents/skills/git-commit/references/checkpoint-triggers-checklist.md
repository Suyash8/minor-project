# Checkpoint Triggers Checklist

Run through this whenever you finish a meaningful chunk of work, to decide whether this
is a commit boundary (Phase 1 of SKILL.md).

- [ ] Did a discrete task or sub-task just complete, and is it verified (not broken,
      not half-done)?
- [ ] Am I about to start something materially different or riskier than what I just
      did, where a clean rollback point would help?
- [ ] Is what I just built a self-contained unit that would be confusing or lossy to
      bundle with whatever comes next?
- [ ] Did the user explicitly ask to commit/save/checkpoint?
- [ ] Is this session/task about to be interrupted, handed off, or is context about to
      be compacted?

**If any box is checked → this is a commit boundary.** Go to Phase 2 (stage
deliberately) — don't skip straight to `git commit -am`.

**If none are checked** → keep working; committing now would likely capture an
incomplete or arbitrary snapshot rather than a real checkpoint.

## Quick sanity checks before actually running the commit

- [ ] `git status` reviewed line by line, not skimmed.
- [ ] No `.env`, credentials, tokens, or private keys among staged files.
- [ ] Staged files all belong to one logical change (split into multiple commits if not).
- [ ] Message describes the change and why, not the mechanism used to make it.
- [ ] Not on a protected branch if this is meant to be a feature branch.
