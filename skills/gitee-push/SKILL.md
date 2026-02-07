---
name: "gitee-push"
description: "Automates and guides the process of pushing a repository to Gitee. Invoke when the user wants to upload, sync, or push code to Gitee."
---

# Gitee Push Skill

This skill assists in pushing the current project to a Gitee repository. It handles initialization, configuration, remote management, and pushing.

## Usage Flow

### 1. Check Repository Status
First, ensure the current directory is a git repository and is clean.
```bash
git status
```
If not initialized:
```bash
git init
```

### 2. Configure Identity (Global or Local)
Gitee requires a valid user identity. If not set, configure it:
```bash
git config --global user.name "Your Name"
```
```bash
git config --global user.email "your.email@example.com"
```
*Tip: You can use your Gitee noreply email (e.g., `digits+username@user.noreply.gitee.com`) for privacy.*

### 3. Commit Changes
Ensure all files are added and committed.
```bash
git add .
git commit -m "Initial commit"
```

### 4. Configure Remote
Add the Gitee repository URL as the `origin` remote.
```bash
git remote add origin https://gitee.com/username/repository.git
```
*If `origin` already exists, check it with `git remote -v` or set a new URL with `git remote set-url origin <url>`.*

### 5. Push to Gitee
Push the master (or main) branch to the remote.
```bash
git push -u origin master
```

## Credential Management (Avoid typing password)

**SECURITY WARNING**: NEVER hardcode passwords in this file or any script.

To save your credentials securely, use the Git credential helper:

```bash
git config --global credential.helper store
```

The next time you push, enter your username and password once. Git will save them locally in `~/.git-credentials`.

## Troubleshooting

- **Authentication Failed**: 
  - Ensure you have the correct username and password.
  - If 2FA is enabled, use a Personal Access Token instead of a password.
  
- **Permission Denied (403)**:
  - Check if you are a member of the repository.
  - Verify if your account or the repo is blocked/suspended.

- **Non-Fast-Forward / Rejected**:
  - If the remote has changes not present locally (e.g., a README created on the web), pull first:
    ```bash
    git pull origin master --allow-unrelated-histories
    ```
