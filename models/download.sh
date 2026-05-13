#!/bin/bash

INPUT_FILE="readme.md"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found."
    exit 1
fi

grep -o 'https://huggingface.co/[^ /]*/[^ /]*' "$INPUT_FILE" | xargs -n 1 -P 1 bash -c '
    REPO_URL="$1"
    DIR_NAME=$(basename "$REPO_URL")

    mkdir -p "$DIR_NAME"
    echo "📂 Processing $DIR_NAME..."

    FILES=(
        "tokenizer.json"
        "tokenizer_config.json"
        "chat_template.jinja"
    )

    for FILE in "${FILES[@]}"; do
        if [[ -f "$DIR_NAME/$FILE" ]]; then
            echo "  ⏩ Skipping $FILE (already exists)"
            continue
        fi
        DL_URL="$REPO_URL/resolve/main/$FILE"
        if curl --output /dev/null --silent --head --fail "$DL_URL"; then
            echo "  ✅ Downloading $FILE -> $DIR_NAME"
            curl -L -s "$DL_URL" -o "$DIR_NAME/$FILE"
        fi
    done
' _
