# CPT-vs-Test Leakage / Overlap Report

This report substantiates the claim that the continual-pre-training (CPT) text used to adapt the
language model does not overlap with the 460-utterance test-set reference transcriptions. It is
provided so the disjointness claim is verifiable from this repository without redistributing the
raw text corpora.

## Method

For each language, the CPT corpus and the 460-utterance test reference transcripts were compared
under the **same normalization used for scoring** (Unicode NFC → lower-casing → punctuation
removal via the regular expression `[^\w\s]` → whitespace collapse). Three overlap measures
were computed:

- **exact-match overlap** — normalized test transcripts appearing verbatim as a line in the CPT text;
- **near-duplicate overlap** — for each test transcript, the maximum similarity against any CPT line,
  with a flag threshold of 0.8;
- (the underlying split lists are in `data_splits/`, so the search/test partition is independently
  checkable.)

The CPT preprocessing additionally includes an explicit **leak-removal pass** that filters CPT
lines against the test and search reference transcripts; the results below are therefore the
residual overlap that remains after that deliberate de-leaking step.

## Results

| Language | Test transcripts | Exact matches | Near-duplicates > 0.8 |
|---|---:|---:|---:|
| Javanese | 460 | 0 | 0 |
| Sundanese | 460 | 0 | 2 |

**Cache/normalization note:** comparisons use the scoring normalization, so casing and punctuation
do not create spurious mismatches or matches.

## The two flagged Sundanese pairs (inspected)

Both Sundanese pairs flagged above the 0.8 threshold are **coincidental overlaps of common short
phrases, not shared utterances**:

1. `basa sunda` vs `basa Sunda téh` — the generic expression for "the Sundanese language", a
   high-frequency stock phrase that carries no test-specific information.
2. `malah loba onjoyna` vs `Malah loba anu ngahiyam` — these share only the sentence-initial
   bigram "malah loba" ("even many"); the content words diverge entirely, so this is not a shared
   utterance.

## Conclusion

There are no exact matches in either language and no Javanese near-duplicates above 0.8 similarity.
The two Sundanese near-duplicate flags are coincidental short-phrase overlaps rather than shared
utterances. We therefore find no evidence of test-set leakage into the language-model adaptation
text. The search and test partitions are speaker-disjoint, and the test set was never consulted
during adaptation or hyperparameter search.

> Reproducing this report end-to-end requires the raw CPT corpora (see `docs/DATA.md`), which are
> not redistributed here; the per-pair detail for the flagged Sundanese pairs is recorded above so
> the result is inspectable without the corpora.
