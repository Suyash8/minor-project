# Download Verification Checklist

Run this immediately after any automated file download (PDF, image, dataset, etc.) and
before logging it as "downloaded" in the source log. A tool call returning without an
error is NOT sufficient evidence the download succeeded.

## Checks, in order

1. **Size sanity check.** A blocked/failed download disguised as a success is very often
   a tiny file (a few hundred bytes to a few KB) — an HTML error page, an access-denied
   page, or a JavaScript bot-challenge page. A genuine paper PDF, image, or dataset is
   almost always much larger. If a file is suspiciously small for its supposed content,
   inspect it before trusting it.

2. **File type / magic-byte check.** Check that the file actually is what its extension
   claims (a real PDF starts with `%PDF`; a real JPEG/PNG has the corresponding magic
   bytes). A `.pdf` file that is actually HTML is the single most common failure mode
   when downloading from publishers with bot protection (MDPI, PMC/NCBI, IEEE Xplore,
   ResearchGate, and similar are known to do this to non-browser HTTP clients).

3. **Skim the content.** Open/read enough of the file to confirm it's the paper/image you
   intended, not a paywall notice, a cookie-consent page, or the wrong version of a
   multi-version preprint.

## If verification fails

1. Delete the bad file — do not leave a corrupt stub sitting in the deliverable folder
   pretending to be real content.
2. Try at most one alternate source if one clearly and legitimately exists: an author's
   own institutional repository, an arXiv/preprint mirror, or a publisher's alternate
   endpoint. Don't spam retries against the same blocked endpoint — if it's bot
   protection, retrying with the same method will keep failing (this is the "retried the
   identical failing approach" anti-pattern — diagnose and switch approach instead).
3. If no working alternative exists, stop trying to force it. Document the failure
   explicitly in the source log: what was attempted, why it failed (if known — e.g.
   "publisher blocks non-browser requests"), and that the source was instead reviewed
   via its online abstract/HTML page. Give the user/reader the direct link so they can
   fetch it manually through a browser if they need the full file.
4. Never fabricate file content to fill the gap, and never claim a source was
   "downloaded" when it was only read online.

## Batch downloads

When downloading many files in one pass (e.g. a loop over several papers):

- Verify every file individually afterward — do not assume that because most succeeded,
  all succeeded. Run the size/type check across the whole batch, not just spot-check one.
- Keep a running tally of which downloads are confirmed-good vs. confirmed-bad vs.
  not-yet-checked, and don't consider the batch done until that tally is fully resolved.
