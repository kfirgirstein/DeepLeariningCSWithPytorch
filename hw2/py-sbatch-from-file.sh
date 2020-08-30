#!/bin/bash
filename=$1
while read line; do
# reading each line
/bin/bash $line
done < $filename


echo "All you jobs are have benn assign~!\n you can watch job status we squeue command."
/usr/bin/watch /usr/bin/squeue
