#!/bin/zsh

files=(*)

for f in $files; do
    if [[ $(echo $f | rev | cut -c1-3 | rev) == "eps" ]]; then
        echo "Converting $f..."
        epspdf --bbox $f
    fi
done

echo "Done."
