#!/bin/bash

echo "Running train split exp"
for train in 0.3 0.4 0.5 0.6 0.7 
  do
  for i in {1..5}
    do
      python main.py --seed $i --iterations 1000 --generations 200 --train-size $train --resume --hidden-size 256
    done
  done

echo "Running iteration exp"
for iteration in 500 1000 5000 10000
  do
  for i in {1..5}
    do
      python main.py --seed $i --iterations $iteration --generations 200 --train-size 0.5 --resume --hidden-size 256
    done
  done

echo "Running Hidden exp"
for hidden in 64 256 512
  do
  for i in {1..5}
    do
      python main.py --seed $i --hidden-size $hidden --iterations 1000 --generations 200 --train-size 0.5 --resume
    done
  done