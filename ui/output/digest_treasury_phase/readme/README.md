# What is in this digest directory

| file | what it is | who reads it |
|---|---|---|
| `digest.json` | all five query result sets in ONE file, plus timings | `board_plan.flatten_digest()` |
| `digest.md` | the same content, rendered to read | a person |
| `revisions.jsonl` etc. | one file per cypher query, one record per line | `head`, `grep`, `wc -l` |
| `readme/*.readme.jsonl` | every field of that query, documented | you, in three months |

Nothing here goes to FLUX. FLUX only ever receives one wordless noun phrase per anchor, built later by `board_plan.glyph_to_prompt()`.
