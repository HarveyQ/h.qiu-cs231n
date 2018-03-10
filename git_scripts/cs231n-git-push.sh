#!/bin/bash
COMMIT_MSG="auto push "
COMMIT_MSG+=`date "+%Y-%m-%d %H:%M:%S"`
cd /Users/HarveyQ/Documents/mlcv/deep_learning/cs231n/cs231n-2017
git add -A
git commit -m "$COMMIT_MSG"
git push origin master

