
import os
import logging
import subprocess
from pyDOE2 import ff2n

logger = logging.getLogger(__name__)


class Main(object):
    def __init__(
        self, k: int, factors: dict, nr_of_replications: int,
        nr_of_experiments: int, models: list
    ):
        self.factors = factors
        self.nr_of_replications = nr_of_replications
        self.nr_of_experiments = nr_of_experiments
        self.models = models
        self.sign_table = ff2n(k)
        self.data = {}
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
                    self.submit_job(
                        self.get_command(
                            cores_level, ram_level, batch_size_level, learning_rate_level,
                            learning_rate_decay_level, model, replication_nr
                        )
                    )

    def get_command(
        self, cores: int, ram: int, batch_size: int,
        learning_rate: float, learning_rate_decay: float, model: str,
        replication_nr, epoch=1
    ):
        command = """ \
        spark-submit --master spark://10.128.0.8:7077 \
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
        --action train \
        --dataPath /tmp/mnist \
        --batchSize {2} \
        --endTriggerNum {3} \
        --learningRate {4} \
        --learningrateDecay {5}
        """.format(cores, ram, batch_size, epoch, learning_rate, learning_rate_decay, model)

        experiment = \
            "{}-Cores, {}-GBRAM, {}-batchSize, {}-learningRate, {}-learningRateDecay" \
            .format(cores, ram, batch_size, learning_rate, learning_rate_decay)

        print("model:", model, ", with parmaters:", experiment)

        if model in self.data:
            if replication_nr in self.data[model]:
                self.data[model][replication_nr].append(experiment)
            else:
                self.data[model][replication_nr] = [experiment]
        else:
            self.data[model] = {replication_nr: [experiment]}

        return command

    def submit_job(self, command: str):
        try:
            # To see output remove stdout and stderr
            FNULL = open(os.devnull, 'w')
            # This type is blocking
            subprocess.check_call(command, shell=True)
            # TODO maybe add another command for running the test to get the accuracy with
            # these hyper-parameters settings after the "train" is called with "test"
        except subprocess.CalledProcessError as e:
            logger.error(e)
            raise


if __name__ == "__main__":
    factors = dict({
        'ram': [1, 8],
        'cores': [1, 4],
        'batch_size': [64, 512],
        'learning_rate': [0.001, 0.1],
        'learning_rate_decay': [0.0001, 0.001]
    })
    models = ['lenet5', 'bi-rnn']
    nr_of_replications = 3
    k = len(factors)
    nr_of_experiments = 2**k  # Full factorial

    main = Main(k, factors, nr_of_replications, nr_of_experiments, models)
    main.run_experiments()

    # Write all replications of all experiments to disc
    print(main.data)
