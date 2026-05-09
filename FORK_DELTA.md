# Fork Delta

This fork carries a small set of intentional deltas from
`NousResearch/hermes-agent`. Keep this file current whenever `origin/main`
contains commits that are not yet in `upstream/main`.

## Current Deltas

| Commit | Classification | Rationale | Next action |
| --- | --- | --- | --- |
| `08a4a8f6ab816abb881327bc3c845ffd57317039` | Fork-only by design | Adds Keegoid operational guidance in `AGENTS.md` so agents use `~/keegoid/ops/bin/agent-pr-flow` for branch, PR, review, and post-merge cleanup work. This path is specific to the Keegoid workspace and should not be sent upstream. | Preserve in the fork during upstream syncs. |
| `1908a8012e8dcf023ad7045a8763dd39f6c6882e` | Upstream-bound | Hardens Discord first voice transcription by warming STT before capture, filtering repeated Whisper hallucinations, logging empty/filtered transcripts, and inferring SSRC before DAVE decrypt. The same patch ID exists on `origin/codex/fix-discord-voice-first-stt-pr` as `425aa7ed115df2cce5e1e2a69882e44ae6ab975b`. | Open or update an upstream PR from a current `upstream/main` base. Keep the fork copy until upstream accepts it, then re-sync without destructive history rewrites unless explicitly approved. |

