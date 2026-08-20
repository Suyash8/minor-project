---
name: deep-research
description: |
  Structured procedure for conducting rigorous research (literature surveys, technical
  investigations, competitive/market analysis, fact-finding) with full source tracking,
  credibility grading, download verification, and a mandatory self-review pass before
  presenting findings. Use this skill whenever the user asks to "research", "survey the
  literature on", "investigate", "look into", "compile sources on", "find out everything
  about", or requests a findings/report document backed by external sources. Do not use
  for simple single-fact lookups that don't require a tracked source trail.
---

# Deep Research

This skill defines how to run a research task end-to-end without fumbling: no invented
sources, no silently-broken downloads, no unverifiable claims, and a paper trail that
survives review. It was written for agents operating with web search, web fetch, a
sandboxed shell, and a filesystem — no special subagent orchestration is required, though
delegating fan-out work to sub-agents is encouraged when available (see Phase 2).

Read this whole file before starting. Do not skip phases under time pressure — the phases
most likely to be skipped (3 and 5) are the ones that catch fabricated or broken sources.

## Phase 0 — Scope the question

Before searching anything:

1. Restate the research question in one or two sentences. If it is broad enough that no
   single report could cover it (e.g. "research X" where X is a whole field), narrow it to
   3-8 concrete sub-topics or use cases and say so explicitly to the user/output, rather
   than silently picking a scope.
2. Decide what "done" looks like: a findings document, an annotated source list, a
   recommendation, a comparison table. Say what artifact you're producing before producing it.
3. If the task implies a deliverable folder or file, create the folder structure first
   (e.g. `papers/`, `images/`, `sources.md`, `findings.md`) so downloads land somewhere
   deterministic instead of scattered in the working directory.
4. Make a todo list (if a todo tool is available) with one item per sub-topic plus items
   for "write source log", "write findings", "verify". Treat research as a multi-step task,
   not a single tool call.

## Phase 1 — Search

1. Run multiple, differently-worded queries per sub-topic (2-4 queries minimum). A single
   query under-samples the literature and biases toward whatever ranks first.
2. Prefer primary sources over secondary summaries when both are available: the original
   paper over a blog post about the paper, the vendor's own docs over a tutorial site.
3. Note every URL you seriously consider, even ones you end up not using — Phase 3 needs
   this list to record what was checked and rejected, not just what was kept.
4. Do not stop at the first plausible-looking result. Cross-reference: if two independent
   sources agree on a claim, it's stronger than one source repeated by several aggregators.

## Phase 2 — Fan-out (optional, for large scope)

If the scoped sub-topics in Phase 0 are independent of each other and a sub-agent /
delegation mechanism is available, dispatch one sub-topic per sub-agent rather than
researching everything serially in one context. Each sub-agent should return: the sources
it consulted, one-paragraph findings per source, and anything it searched for but could
not find. Treat sub-agent output the same as your own Phase 1 results — still subject to
Phase 3 verification, not blindly trusted.

If no delegation mechanism is available, work sub-topics one at a time and keep the
per-sub-topic notes separate before merging, so a mistake in one sub-topic doesn't
contaminate the others.

## Phase 3 — Verify before you cite or download (this is the phase people skip)

For every source you intend to actually use:

1. **Confirm it says what you think it says.** Fetch the actual page/abstract, don't rely
   solely on a search-result snippet — snippets truncate and can flip meaning.
2. **Check freshness.** If the topic is time-sensitive (versions, prices, current events,
   "latest"), note the source's publish date and flag anything stale.
3. **If downloading a file (PDF, image, dataset):**
   - Download it, then immediately check the result. A failed or blocked download often
     *looks* successful (HTTP 200, a file gets written) but contains an HTML error page,
     a bot-challenge page, or a redirect notice instead of the real content.
   - Concretely: check file size (a real paper PDF is rarely under ~50KB; a blocked
     download is often a few hundred bytes) and check the actual file type/magic bytes,
     not just the extension you gave it.
   - If a publisher blocks automated downloads (common with MDPI, PMC/NCBI, IEEE, and
     similar bot-protected hosts), do not fake it or leave a corrupt file in place.
     Delete the bad file, try one alternate mirror if one clearly exists (e.g. an
     author's institutional repository, arXiv, a preprint server), and if that also
     fails, **say so explicitly** in the source log instead of silently omitting it or
     pretending it downloaded. A documented gap is fine; a silent one is not.
4. **Check licensing before reusing anything (images, datasets, code snippets).** Only
   use assets with a clear license, and record what that license is and what attribution
   it requires. When in doubt, don't use it.
5. **Never fabricate a URL, author name, date, or statistic.** If you cannot verify a
   detail, mark it `[unverified]` in your notes rather than smoothing it over with a
   plausible-sounding guess. An agent that says "I couldn't confirm X" is more trustworthy
   than one that never admits uncertainty.

## Phase 4 — Log every source as you go

Do not wait until the end to reconstruct what you looked at — you will forget or
misattribute things. Maintain a running source log (a single markdown file is usually
enough) with, at minimum, per source:

- The link.
- What you did with it: downloaded (and the exact saved filename/path), read online only,
  or considered-but-rejected (and why).
- A one-paragraph, in-your-own-words summary of its actual content (not a copy-paste of
  the abstract — light paraphrase plus a compliance note if you lean on their wording).
- What you personally took away from it / how it bears on the research question.

Group the log by sub-topic/use-case, not by the order you happened to find things — a
log ordered by discovery order is hard for anyone (including future-you) to use.

## Phase 5 — Self-review before presenting (mandatory, do not skip)

Before writing the final findings/report:

1. Re-read your own source log. For each major claim in your draft findings, confirm it
   traces back to a specific logged source. Delete or soften any claim that doesn't.
2. Actively look for disagreement between sources, not just confirmation. If two sources
   conflict, say so rather than quietly picking the one that fits your narrative.
3. Check completeness against Phase 0's scope: did you actually cover every sub-topic you
   committed to, or did some quietly get dropped along the way?
4. Check the deliverable folder/files actually exist and are non-empty/non-corrupt (see
   Phase 3, point 3) — verify with a directory listing and file-size check, don't assume a
   tool call succeeded just because it returned without an error.
5. State explicitly what you verified firsthand versus what you're reporting from a
   secondary source you couldn't independently confirm.

## Phase 6 — Deliver

Produce the artifact decided in Phase 0. At minimum this typically means:

- A **source log** (per Phase 4) — the paper trail.
- A **findings/synthesis document** — organized by topic/use-case, not by chronological
  search order, that answers the original question and cites the source log.
- Any downloaded files, in a predictable folder structure, with filenames that make it
  obvious what each one is without opening it.

## Anti-patterns (things that constitute "fumbling")

- Treating a search-result snippet as verified fact without opening the source.
- Writing a source log after the fact from memory instead of as you go.
- Leaving a corrupted/blocked download in place without noticing or flagging it.
- Citing a source for a claim it doesn't actually support.
- Silently narrowing scope mid-task without telling the user.
- Presenting "I searched and found nothing" the same way as "I verified this doesn't
  exist" — these are different claims with different confidence levels.
- Re-trying the exact same failed approach twice instead of diagnosing why it failed
  (e.g. retrying a blocked download with the same method, or a shell command with the
  wrong shell syntax) — switch approach after one diagnosed failure, not after several
  identical ones.
- Skipping Phase 5 because the draft "looks fine."

## Reference files

- `references/source-credibility-rubric.md` — how to grade a source's trustworthiness.
- `references/source-log-template.md` — the minimum structure for Phase 4's log.
- `references/download-verification-checklist.md` — the exact checks for Phase 3.3.
