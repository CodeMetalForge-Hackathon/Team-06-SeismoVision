# Team collaboration

All four teammates have admin access to the private [Team-6 repository](https://github.com/CodeMetalForge-Hackathon/Team-6). GitHub is the shared file history and submission location. Each teammate works in their own local clone and Codex chat.

## Clone and open in Codex

Use one of the following clone commands:

```bash
git clone git@github.com:CodeMetalForge-Hackathon/Team-6.git
```

```bash
git clone https://github.com/CodeMetalForge-Hackathon/Team-6.git
```

Open the cloned `Team-6` folder as a local Codex project. Start a new Codex chat from that folder. Codex discovers the checked-in `AGENTS.md`, README, and project docs there. Each person has their own chat transcript and working files; the GitHub repository carries shared changes. A shared Codex chat link is a [read-only snapshot](https://learn.chatgpt.com/docs/use-chatgpt), not a jointly editable chat or a shared local folder.

## Make a change

1. Run `git pull --ff-only origin main` before starting.
2. Create a focused branch with `git switch -c feature/short-description`.
3. Commit work with a message describing what changed. The event judges count work done during the official build window, so preserve clear Git history.
4. Push with `git push -u origin feature/short-description` and open a pull request on GitHub. Ask a teammate to review when time allows.
5. Merge the pull request, then update local `main` with `git switch main` and `git pull --ff-only origin main`.

Coordinate urgent changes to `main` in person before pushing so a teammate's work is not overwritten. Never force-push shared branches. The implementation owner chooses the code layout and documents exact setup and verification commands in `code/README.md` when their work lands.

## Keep the handoff usable

- Put project decisions in [the project brief](docs/project/earthquake-stability.md), experimental results in [the demo runbook](docs/demo/runbook.md), and sources in [research and evidence](docs/research/evidence.md).
- The [Google Doc](docs/brainstorm/README.md) remains the live brainstorming space; reconcile durable conclusions into Git before submission.
- Label recorded or simulated demo data clearly. Do not present a recorded replay as a live measurement.
- Push all intended submission code to the team repo before **12:00 p.m. Eastern on November 8, 2026**.

Sources for Codex behavior: [Projects and chats](https://learn.chatgpt.com/docs/projects) and [sharing Codex chats](https://learn.chatgpt.com/docs/use-chatgpt).
