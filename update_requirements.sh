#!/bin/bash
REQ_FILE="requirements.txt"
HASH_FILE=".requirements_hash"

# Get hash of current installed package list
CURRENT_HASH=$(pip list --format=freeze --not-required | sha256sum | cut -d' ' -f1)

# If no hash file or hash changed, update requirements.txt
if [ ! -f "$HASH_FILE" ] || [ "$CURRENT_HASH" != "$(cat "$HASH_FILE")" ]; then
    pip list --format=freeze --not-required > "$REQ_FILE"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "[requirements.txt updated]"
else
    echo "[No package changes detected]"
fi

