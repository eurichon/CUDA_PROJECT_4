for i in 2048 4096 8192 16384 32768 65536
do
	for j in 10 100 1000 10000
	do 
		./tests/test_distance $i $j
	done
done
