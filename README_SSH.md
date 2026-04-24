# SSH Notes for This Repo

This repo is often edited from a Docker container path under WSL.  
Git/SSH always uses the key from the environment where `git push` is executed.

## What to remember

- If you push from inside the container, it uses the container user's `~/.ssh`.
- If you push from WSL shell, it uses WSL user's `~/.ssh`.
- A key existing in WSL does not help container SSH unless shared/mounted.

## Common failure symptoms

- `ssh -T git@github.com` hangs or fails with `Permission denied (publickey)`.
- `ssh -v -T git@github.com` shows `identity file ... type -1`.
- `ls -la ~/.ssh` only shows `known_hosts` and no private key.

## Quick recovery (inside current runtime)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Then add the printed public key to GitHub SSH keys and verify:

```bash
ssh -T git@github.com
git push origin master
```

## Optional sanity checks

```bash
whoami
echo "$HOME"
ls -la ~/.ssh
```

These confirm which environment and SSH home are actually active.
