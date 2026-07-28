# Triage Labels

The skills speak in terms of five canonical triage states. This file maps those states to the actual YAML frontmatter values used in this repo's local markdown issues.

| Label in skills | Value in our tracker | Meaning                                  |
| --------------- | -------------------- | ---------------------------------------- |
| `needs-triage`  | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`    | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent` | `ready-for-agent`  | Fully specified, ready for an AFK agent  |
| `ready-for-human` | `ready-for-human`  | Requires human implementation            |
| `wontfix`       | `wontfix`            | Will not be actioned                     |

When a skill mentions a state (e.g. "apply the AFK-ready triage state"), use the corresponding string from this table.

Categories are separate from these states. The default categories are `bug`, `enhancement`, and `research`.
