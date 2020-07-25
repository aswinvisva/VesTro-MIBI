#!/bin/sh

echo "You are about to commit on master. I will run your tests first..."
python -m unittest discover tests

    if [ $? -eq 0 ]; then
        # tests passed, proceed to prepare commit message
        git add .
        git status
        read -p "Commit description: " desc
        git commit -m $desc
        git push origin master
        exit 0
    else
        # some tests failed, prevent from committing broken code on master
        echo "Some tests failed. You are not allowed to commit broken code on master! Aborting the commit."
        echo "Note: you can still commit broken code on feature branches"
        exit 1
    fi

