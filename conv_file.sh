#!/bin/bash
filename='wrd_instances.csv'
echo Start
while read p; do 
	a=${p%,*}
	echo $a >> entities.txt
done < $filename
