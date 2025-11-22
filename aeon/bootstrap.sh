#!/bin/bash
# Env setup that must be done before any uv commands
# Install rust so that huggingface install works
command -v rustc >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

