#!/bin/bash
COMMIT_MSG="auto push "
COMMIT_MSG+=`date "+%Y-%m-%d %H:%M:%S"`

cd ~/cs231n
git add -A
git commit -m "$COMMIT_MSG"
git push origin master

