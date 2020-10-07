# qpecs-project


# Setup the history server

Create a directory `spark-events` in your home directory

Append the following lines to two files `/home/am72ghiassi/bd/spark/conf/spark-defaults.conf` and `/home/am72ghiassi/bd/spark/conf/spark-bigdl.conf`

replace `<username>` with your username

```
spark.eventLog.enabled                      true
spark.eventLog.dir                          file:///home/<username>/spark-events
spark.history.fs.logDirectory               file:///home/<username>/spark-events
```

After doing those steps run the following command to start history-server:

```
start-history-server.sh
```
and then you can run the script to get the job status

