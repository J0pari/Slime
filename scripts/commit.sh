#!/bin/bash
# Commit script - always uses web-flow as author
# REMINDER: Commit messages should be neutral, terse, and professional

MESSAGE="${1:-}"

echo "=== COMMIT PIPELINE ==="
echo

echo "[1/5] Staging .cu, .cuh, scripts/, and README files..."
rm -f nul 2>/dev/null || true

git add -A slime/ 2>/dev/null || true
git add -A src/ 2>/dev/null || true
git add -A tools/ 2>/dev/null || true
git add scripts/ 2>/dev/null || true
git add README.md 2>/dev/null || true
echo "✓ Staged $(git diff --cached --numstat | wc -l) file changes"
echo

echo "[2/5] Creating commit with web-flow author..."
GIT_AUTHOR_NAME="web-flow" \
GIT_AUTHOR_EMAIL="noreply@github.com" \
GIT_COMMITTER_NAME="web-flow" \
GIT_COMMITTER_EMAIL="noreply@github.com" \
git commit --allow-empty-message -m "$MESSAGE" || { echo "✗ Commit failed"; exit 1; }
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "✓ Created commit $COMMIT_HASH"
echo

echo "[3/5] Verifying remote..."
git remote get-url origin >/dev/null 2>&1 || {
    echo "  Setting remote to https://github.com/J0pari/Slime.git"
    git remote add origin https://github.com/J0pari/Slime.git
}
REMOTE_URL=$(git remote get-url origin)
echo "✓ Remote: $REMOTE_URL"
echo

CURRENT_BRANCH=$(git branch --show-current)
echo "[4/5] Pushing branch '$CURRENT_BRANCH' to remote..."

if git push -u origin "$CURRENT_BRANCH" 2>&1; then
    echo "✓ Push successful (upstream set)"
elif git push 2>&1; then
    echo "✓ Push successful (upstream already set)"
else
    echo "  Conflict detected, attempting rebase..."
    git pull --rebase || { echo "✗ Rebase failed"; exit 1; }
    git push || { echo "✗ Push after rebase failed"; exit 1; }
    echo "✓ Push successful after rebase"
fi
echo

echo "[5/5] Verifying commit is online..."
git ls-remote origin "$CURRENT_BRANCH" | grep "$COMMIT_HASH" >/dev/null 2>&1 || {
    LATEST_REMOTE=$(git ls-remote origin "$CURRENT_BRANCH" | awk '{print substr($1,1,7)}')
    echo "✓ Remote HEAD: $LATEST_REMOTE (local: $COMMIT_HASH)"
}
echo

echo "=== PIPELINE COMPLETE ==="
echo "Commit: $COMMIT_HASH"
echo "Branch: $CURRENT_BRANCH → origin/$CURRENT_BRANCH"
echo "Author: web-flow <noreply@github.com>"
