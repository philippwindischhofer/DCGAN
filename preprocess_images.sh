#!/bin/bash

SIZESTRING="128X128"

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

CURRENT_DIR=`pwd`

mkdir -p $OUT_DIR

cd $IN_DIR

PIC_LIST=`ls`

COUNT=0

for PIC in $PIC_LIST
do
    eval convert $IN_DIR$PIC -thumbnail "'"${SIZESTRING}">'" -background black -gravity center -extent $SIZESTRING $OUT_DIR$COUNT.png
    COUNT=`expr $COUNT + 1`
done
