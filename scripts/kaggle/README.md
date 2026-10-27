# Kaggle GPU jobs

Run jobs on a free Kaggle T4 GPU from the training PC. A job is packed into one private notebook, pushed
with the official Kaggle API, and its report downloaded when it finishes.

```
fly kaggle setup      # once: your Kaggle username + API key, saved on the PC only
fly kaggle mjx        # push the MJX feasibility test
fly kaggle status     # running / complete
fly kaggle pull       # download the report and print the summary
```

## Keys and security (the repo is public)

- The key is kept **outside the repo** in `~/.config/flybrain/secrets.env`, readable only by you (600).
  `fly kaggle setup` asks for it with hidden typing, or reads a downloaded `kaggle.json` and then deletes it.
- It is never written into code, never passed on a command line, and only given to the `kaggle` program.
- `pack.py` packs only an allow-list of code files, and refuses to write anything that contains your key or
  looks like an API key. The notebook is private, and the job on Kaggle needs no key.
- A git pre-commit hook (installed by `fly kaggle setup`) blocks commits with the key or a `.env` /
  `kaggle.json` file; `.gitignore` excludes them too.
- `fly kaggle forget` deletes the saved key. If a key ever leaks, revoke it on kaggle.com > Settings > API.

## Jobs

`mjx`: `mjx_feasibility.py`. Can the phase 2c scene run on a GPU with MuJoCo MJX? CPU baseline, MJX speed
for 64-4096 arms (scene as-is and with arm-link collisions off), both cameras as GPU ray casts, and a
physics check (the teacher's CPU episode replayed in MJX). Local check on a CPU: `--quick`.
