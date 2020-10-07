#!/bin/bash

#This function checks if all jobs was succeeded;
#If one of the jobs is failed the result will be failed
function allSucceeded {
	result="failed"
	jobs=$(grep "Job Result" "$1" | wc -l)
	jobsSucceeded=$(grep "JobSucceeded" "$1" | wc -l)
	[[ $jobs -gt 0 ]] && [[ $jobs -eq $jobsSucceeded ]] && result="succeeded"
	echo $result
}

#This function checks if the last jobs was succeeded;
function lastSucceeded {
	result="failed"
	jobsSucceeded=$(grep "Job Result" "$1" | tail -1 | grep -o "\"Result\":\"JobSucceeded\"" | wc -l)
	[[ $jobsSucceeded -gt 0 ]] && result="succeeded"
	echo $result
}

#############CALL VICTOR SCRIPT TO RUN THE MODEL#########################
TIMEFORMAT=%3R
eta=$(time (/home/am72ghiassi/bd/spark/bin/runlenet.sh 2>&1>/dev/null) 2>&1>/dev/null)
unset TIMEFORMAT
#############CALL VICTOR SCRIPT TO RUN THE MODEL#########################

#Get the last file created in spark-events directory
app=$(ls -td -- ~/spark-events/* | head -1)

#Check if the model was succeeded
res=$(lastSucceeded "$app")
#res=$(allSucceeded "$app")

#print the result (succeeded | failed)
echo $res
[[ "$res" = "failed" ]] && eta=107374182
echo $eta
