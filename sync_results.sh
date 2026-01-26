#!/bin/bash
# Script to sync simulation results from /tmp/hybrid_control to git repository
# and commit/push changes

set -e  # Exit on error

# Configuration
SOURCE_DIR="/tmp/hybrid_control"
REPO_DIR="/home/docker_user/catkin_ws/src/contact_maintain"
TARGET_DIR="${REPO_DIR}/results/hybrid_control"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Syncing Results to Git Repository ===${NC}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Error: Source directory $SOURCE_DIR does not exist${NC}"
    exit 1
fi

# Check if repo directory exists
if [ ! -d "$REPO_DIR" ]; then
    echo -e "${RED}Error: Repository directory $REPO_DIR does not exist${NC}"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Change to repo directory
cd "$REPO_DIR"

# Sync files (copy with update flag - only newer files)
echo -e "${YELLOW}Copying files from $SOURCE_DIR to $TARGET_DIR...${NC}"
if ls "$SOURCE_DIR"/* 1> /dev/null 2>&1; then
    cp -u "$SOURCE_DIR"/* "$TARGET_DIR/" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Some files may not have been copied${NC}"
    }
    echo -e "${GREEN}Files copied successfully${NC}"
else
    echo -e "${YELLOW}Warning: No files found in $SOURCE_DIR${NC}"
fi

# Check git status
echo -e "${YELLOW}Checking git status...${NC}"
git status --short results/hybrid_control/ > /tmp/git_status.txt || true

if [ ! -s /tmp/git_status.txt ]; then
    echo -e "${GREEN}No changes to commit. All files are up to date.${NC}"
    rm -f /tmp/git_status.txt
    exit 0
fi

# Show what will be committed
echo -e "${YELLOW}Files to be committed:${NC}"
cat /tmp/git_status.txt
rm -f /tmp/git_status.txt

# Ask for commit message (or use default)
if [ -z "$1" ]; then
    COMMIT_MSG="Sync hybrid_control results: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}Using default commit message:${NC} $COMMIT_MSG"
else
    COMMIT_MSG="$1"
    echo -e "${YELLOW}Using provided commit message:${NC} $COMMIT_MSG"
fi

# Add files
echo -e "${YELLOW}Staging files...${NC}"
git add results/hybrid_control/

# Commit
echo -e "${YELLOW}Committing changes...${NC}"
git commit -m "$COMMIT_MSG" || {
    echo -e "${RED}Error: Commit failed${NC}"
    exit 1
}

# Push
echo -e "${YELLOW}Pushing to remote...${NC}"
git push origin master || {
    echo -e "${RED}Error: Push failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Successfully synced and pushed results to git repository!${NC}"
