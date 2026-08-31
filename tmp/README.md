# Transcript slices for testing external tools

Extracted from the GroupMemBench corpora and kept here deliberately: these are the
exact inputs we fed to third-party tools, so results stay reproducible and
comparable. Regenerating them from a script is not equivalent — the selection
heuristic changes, and then a later comparison is no longer like-for-like.

| File | Domain · channel | Size | What it contains |
|---|---|---|---|
| `01_baseline_regulatory_compliance.txt` | Finance · Regulatory Compliance Program | 14 msg · 1,063 w | Ordinary discussion, no reversal. **Fed to sketchnote.app 2026-08-31** → `modules/kg-agent-memory/figures/sketchnote_strategic_project_alignment_framework.jpg` |
| `02_progression_compliance_review_gate.txt` | Finance · Customer Onboarding | 25 msg · 1,512 w | The gate moves 63 % → 74 % while "June 28" is re-negotiated. A *progression*, not a reversal. |
| `03_reversal_finance_aml_three_bucket_split.txt` | Finance · AML Project | 30 msg · 1,632 w | ⭐ The team commits to a three-bucket split, then abandons it: *"the original split is no longer safe"* → *"pause scenario creation"*. |
| `04_reversal_technology_backup_owners.txt` | Technology · NotificationAgent | 30 msg · 1,984 w | ⭐ Explicit pushback: *"I'd push back on backup owners this early"* → settles on one primary signer per signal. |

## How the reversal slices were found

Not by grepping for keywords. GroupMemBench's **`knowledge_update` gold answers**
state what the *current* approach is, and most phrase it as a contrast with what
came before ("Instead of …", "rather than …", "no longer …"). That makes the
question set a human-curated index of every supersession in the corpus.

`analysis/find_reversals.py` BM25-matches each such answer back against the
messages and reports where the evidence sits:

```bash
cd modules/kg-agent-memory
./.venv/bin/python analysis/find_reversals.py               # all four domains
./.venv/bin/python analysis/find_reversals.py Technology    # one domain
```

Reversal density by domain:

| Domain | knowledge_update questions | stating a reversal |
|---|---:|---:|
| **Finance** | 32 | **20 (62 %)** |
| Technology | 36 | 19 (53 %) |
| Manufacturing | 22 | 8 (36 %) |
| Healthcare | 17 | 1 (6 %) |

Finance is the most contentious domain — convenient, since it is the one already
ingested. **Healthcare is nearly reversal-free** and would be a poor choice for
demonstrating supersession.
