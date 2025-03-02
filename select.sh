#!/bin/bash

# Define source and destination directories
SOURCE_DIR="seed-images"
DEST_DIR="selected-seeds"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Declare an associative array mapping seed numbers to names
declare -A SEED_MAP=(
    [147]="deep blue"
    [205]="pink sea"
    [308]="sunset beach"
    [353]="sunset beach"
    [357]="sunset beach smooth"
    [389]="pink beach sand"
    [425]="clear beach"
    [427]="beach"
    [472]="pink night beach"
    [485]="sand beach"
    [504]="pink sunset sand beach"
    [533]="coral beach"
    [551]="waves"
    [559]="sunset beach"
    [565]="pink mountains"
    [583]="rocky beach"
    [615]="sunrise beach"
    [616]="sunrise beach"
    [621]="sunset beach"
    [650]="rocky beach"
    [652]="grey sea"
    [670]="dream beach"
    [674]="sea"
    [690]="rocky beach"
    [694]="sea closeup"
    [698]="grey sea"
    [700]="pink mountains"
    [703]="rocky beach"
    [706]="sunset sandy beach"
    [715]="big sun beach"
    [718]="sunset beach"
    [719]="pink mountain"
    [765]="pink sunset"
    [794]="sandy beach"
    [800]="red sunset"
    [806]="grey mountain smooth"
    [812]="sunset sea"
    [813]="purple forest"
    [815]="purple sunset"
    [820]="sunrise sandy beach"
    [827]="close sea"
    [840]="far sea"
    [845]="beautiful yellow blue sea"
    [853]="sunset sea"
    [894]="sunrise sea"
    [902]="sunrise sea"
    [934]="red mountains"
    [947]="sea"
    [953]="sunset sea"
    [954]="vague sea"
    [984]="pink sunrise"
    [994]="red sun sea"
)

# Copy and rename files
for SEED in "${!SEED_MAP[@]}"; do
    SRC_FILE="$SOURCE_DIR/seed_${SEED}_first.tiff"
    DEST_FILE="$DEST_DIR/${SEED} - ${SEED_MAP[$SEED]}.tiff"

    if [[ -f "$SRC_FILE" ]]; then
        cp "$SRC_FILE" "$DEST_FILE"
        echo "Copied $SRC_FILE to $DEST_FILE"
    else
        echo "Warning: $SRC_FILE not found."
    fi
done

echo "Copy and renaming process completed."