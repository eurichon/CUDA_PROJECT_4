for i in 1024 2048 4096 8192 16384 32768 65536
do
	for j in 10 100 1000 5000
	do 
		./tests/vptree $i $j
	done
done
