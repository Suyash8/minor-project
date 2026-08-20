# Commit Message Examples

## Good

```
Add literature survey on object detection for AVs

Adds sources.md (12 papers + 3 licensed images logged with summaries
and takeaways) and findings.md (synthesis across six narrowed use
cases: 2D detection, 3D/LiDAR, sensor fusion, adverse weather, VRU
detection, edge deployment).
```

```
Set up deep-research agent skill and AGENTS.md
```

```
Fix broken PDF downloads in download-verification-checklist

MDPI/PMC block non-browser HTTP clients and return HTML stub pages
instead of the PDF. Added an explicit size/magic-byte check step so
this gets caught before being logged as a successful download.
```

```
WIP: draft findings.md outline, sections 1-3 only
```
(only acceptable when explicitly checkpointing unfinished work, and
labeled as such)

## Bad, and why

```
update files
```
No information about what changed or why. Useless in `git log` six months from now.

```
Ran several tool calls to add stuff to the papers folder and also fixed
the steering file and also updated AGENTS.md and also renamed a few things
```
Describes the mechanism (tool calls) instead of the change, and bundles three unrelated
concerns that should have been three separate commits.

```
fix
```
No context. If this is truly the only reasonable summary, the change was described too
vaguely to know if it was scoped correctly in the first place.

```
Committing progress as requested by user, this commit adds the following
files: [long exhaustive line-by-line list of every file with no synthesis]
```
A message shouldn't restate `git diff --stat`; explain intent, let `git show` show the
diff itself.
