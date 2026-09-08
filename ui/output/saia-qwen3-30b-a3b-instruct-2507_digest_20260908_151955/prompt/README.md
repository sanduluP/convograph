# The prompt this run sent to the board planner

| file | what it is |
|---|---|
| `full.txt` | every message in order — what actually went over the wire |
| `system.txt` | the standing instructions (`prompts/board_plan_system.txt`) |
| `user.txt` | this run's input: the meeting messages, then the indexed facts |
| `reply_raw.txt` | the model's reply before JSON extraction |

Total prompt: **11,937 characters** (~2,984 tokens).

The fact indices in `user.txt` are the same ones `plan.json` refers to in `anchors[].from_facts`, `notes[].fact` and `dropped`, so a board element can be traced to the exact line the planner read.
