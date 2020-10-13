
import os
import json
import logging
import subprocess
import sys
import time
from pyDOE2 import ff2n
from optparse import OptionParser


logger = logging.getLogger(__name__)


class Main(object):
    def __init__(
        self, k: int, factors: dict, nr_of_replications: int,
        nr_of_experiments: int, models: list, master: str
    ):
        self.factors = factors
        self.nr_of_replications = nr_of_replications
        self.nr_of_experiments = nr_of_experiments
        self.models = models
        self.sign_table = ff2n(k)
        self.data = {}
        self.master = master
        super().__init__()

    def run_experiments(self):
        for model in self.models:  # for each model
            for replication_nr in range(self.nr_of_replications):  # do n replications
                for i in range(len(self.sign_table)):  # rows
                    cores_level = self.factors['cores'][-1 if int(self.sign_table[i][0]) == -1 else 0]
                    ram_level = self.factors['ram'][-1 if int(self.sign_table[i][1]) == -1 else 0]
                    batch_size_level = self.factors['batch_size'][-1 if int(self.sign_table[i][2]) == -1 else 0]
                    learning_rate_level = self.factors['learning_rate'][-1 if int(self.sign_table[i][3]) == -1 else 0]
                    learning_rate_decay_level = self.factors['learning_rate_decay'][-1 if int(self.sign_table[i][4]) == -1 else 0]
                    print("New Job Started", model, "replication: ", (replication_nr + 1), "/", self.nr_of_replications, "sing table: ", (i+1), "/", len(self.sign_table));
                    status, accuracy, responsetime = self.submit_job(
                            cores_level, ram_level, batch_size_level, learning_rate_level,
                            learning_rate_decay_level, model, replication_nr
                    )
                    self.save_result(cores_level, ram_level, batch_size_level, learning_rate_level,
                            learning_rate_decay_level, model, replication_nr, status, accuracy, responsetime)

    def get_command(
        self, cores: int, ram: int, batch_size: int,
        learning_rate: float, learning_rate_decay: float, model: str,
        replication_nr, action: str, modelPath: str, epoch=1
    ):
        command = """ \
        spark-submit --master spark://{7}:7077 \
        --driver-cores {0} \
        --driver-memory {1}G \
        --total-executor-cores {0} \
        --executor-cores {0} \
        --executor-memory {1}G \
        --py-files /home/am72ghiassi/bd/spark/lib/bigdl-0.11.0-python-api.zip,/home/am72ghiassi/bd/codes/{6}.py \
        --properties-file /home/am72ghiassi/bd/spark/conf/spark-bigdl.conf \
        --jars /home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar \
        --conf spark.driver.extraClassPath=/home/am72ghiassi/bd/spark/lib/bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar \
        --conf spark.executer.extraClassPath=bigdl-SPARK_2.3-0.11.0-jar-with-dependencies.jar /home/am72ghiassi/bd/codes/{6}.py \
        --action {8} \
        --dataPath /tmp/mnist \
        --checkpointPath \"/tmp/{6}\" \
        --modelPath {9} \
        --batchSize {2} \
        --endTriggerNum {3} \
        --learningRate {4} \
        --learningrateDecay {5} \
        """.format(cores, ram, batch_size, epoch, learning_rate, learning_rate_decay, model, self.master, action, modelPath)


        return command

    def submit_job(
        self, cores: int, ram: int, batch_size: int,
        learning_rate: float, learning_rate_decay: float, model: str,
        replication_nr, epoch=1
            ):
        status = "failed"
        accuracy = 0.0
        responsetime = 107374182
        try:
            # To see output remove stdout and stderr
            FNULL = open(os.devnull, 'w')
            train_command = self.get_command(
                    cores, ram, batch_size, learning_rate, learning_rate_decay, model, replication_nr, "train", "")
            # This type is blocking
            start = time.perf_counter()
            subprocess.check_call(train_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
            end = time.perf_counter()
            status = "succeeded"
            responsetime = (end-start)
            #Find the train model directory
            modelPath_command = "ls -td -- /tmp/{}/*/model.* | head -1".format(model)
            modelPath = subprocess.check_output(modelPath_command, shell=True, stderr=FNULL).decode("utf-8").rstrip()
            # these hyper-parameters settings after the "train" is called with "test"
            print("Running The test with: ", modelPath)
            test_command = "{} | grep -o -P \"result: \K[0-9\.]*\"".format(
                    self.get_command(
                    cores, ram, batch_size, learning_rate, learning_rate_decay, model, replication_nr, "test", modelPath)
                    )
            accuracy_result = subprocess.check_output(test_command , shell=True, stderr=FNULL).decode("utf-8").rstrip()
            if (accuracy_result != ""):
                accuracy = accuracy_result
        except subprocess.CalledProcessError as e:
            pass
            #logger.error(e)
            #raise

        # clean work folder after each job
        os.system("rm -rf /home/am72ghiassi/bd/spark/work/* &>/dev/null")
        os.system("rm -rf ~/spark-events/* &>/dev/null")
        return status, accuracy, responsetime,

    def save_result(
        self, cores: int, ram: int, batch_size: int,
        learning_rate: float, learning_rate_decay: float, model: str,
        replication_nr, status: str, accuracy: str, responsetime: float
        ):

        experiment = \
            "{}-Cores, {}-GBRAM, {}-batchSize, {}-learningRate, {}-learningRateDecay, {}-status, {}-accuracy, {}-responsetime" \
            .format(cores, ram, batch_size, learning_rate, learning_rate_decay, status, accuracy, responsetime)

        print("model:", model, ", with parmaters:", experiment)

        if model in self.data:
            if replication_nr in self.data[model]:
                self.data[model][replication_nr].append(experiment)
            else:
                self.data[model][replication_nr] = [experiment]
        else:
            self.data[model] = {replication_nr: [experiment]}



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--master", dest="master", default="10.128.0.5")

    (options, args) = parser.parse_args(sys.argv)

    factors = dict({
        'ram': [1, 8],
        'cores': [1, 4],
        'batch_size': [64, 512],
        'learning_rate': [0.001, 0.1],
        'learning_rate_decay': [0.0001, 0.001]
    })
    models = ['bi-rnn', 'lenet5']
    nr_of_replications = 3
    k = len(factors)
    nr_of_experiments = 2**k  # Full factorial

    main = Main(k, factors, nr_of_replications, nr_of_experiments, models, options.master)
    main.run_experiments()

    # Write all replications of all experiments to disc
    with open('data.json', 'w') as fp:
        json.dump(main.data, fp, sort_keys=True, indent=4)
