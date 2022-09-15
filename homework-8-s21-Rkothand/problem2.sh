#!/usr/bin/env bash
new_file=${OUTFILE}.out
echo $new_file
err_file=${OUTFILE}.err
echo $err_file
./cmd1 < $INFILE | ./cmd3 > $new_file 2>$err_file
