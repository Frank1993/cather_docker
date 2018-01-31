#!/bin/bash

cwd=$(dirname "${BASH_SOURCE[0]}")

if [[ -f ~/.bash_history ]]; then
    cp ~/.bash_history $cwd/bash_history
fi
