#!/bin/bash

CUR_DIR=`pwd`
SIZESTRING="64x64"

POSARG=()

while [[ $# -gt 0 ]]
do
key=$1

case $key in
    --in)
    IN_DIR="$2/"
    shift
    shift
    ;;
    --out)
    OUT_DIR="$2/"
    shift
    shift
    ;;
    *)
    POSARG+=("$1")
    shift
    ;;
esac
done

set -- "${POSARG[@]}"

mkdir -p $OUT_DIR

PIC_LIST=`ls $IN_DIR`

COUNT=0

for PIC in $PIC_LIST
do
    echo "writing" $OUT_DIR$COUNT.png

    convert $IN_DIR$PIC -thumbnail $SIZESTRING -background black -gravity center -extent $SIZESTRING $OUT_DIR$COUNT.png
    convert $OUT_DIR$COUNT.png -alpha off $OUT_DIR$COUNT.png

    COUNT=`expr $COUNT + 1`
done
