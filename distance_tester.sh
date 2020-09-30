#!/bin/bash
for i in  20000
do
	for j in 1024 2048 4096 8192 16384 32768 
	do
		./tests/test_distance $j $i
	done
done
