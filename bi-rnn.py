import matplotlib
matplotlib.use('Agg')


import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from utils import get_mnist
from pyspark import SparkContext
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from optparse import OptionParser
from bigdl.dataset import mnist
from bigdl.dataset.transformer import *
from bigdl.optim.optimizer import *





# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

def build_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = BiRecurrent(JoinTable(3, 3))
    recurrent.add(LSTM(input_size, hidden_size))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Linear(2*hidden_size, output_size))
    return model
#rnn_model = build_model(n_input, n_hidden, n_classes)

# Create an Optimizer
def get_mnist(sc, data_type="train", location="/tmp/mnist"):
    """
    Get and normalize the mnist data. We would download it automatically
    if the data doesn't present at the specific location.

    :param sc: SparkContext
    :param data_type: training data or testing data
    :param location: Location storing the mnist
    :return: A RDD of (features: Ndarray, label: Ndarray)
    """
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1) # Target start from 1 in BigDL
    record = images.zip(labels)
    return record

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-o", "--modelPath", dest="modelPath", default="/tmp/bi-rnn/model.470")
    parser.add_option("-c", "--checkpointPath", dest="checkpointPath", default="/tmp/bi-rnn")
    parser.add_option("-t", "--endTriggerType", dest="endTriggerType", default="epoch")
    parser.add_option("-n", "--endTriggerNum", type=int, dest="endTriggerNum", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")
    parser.add_option("-l", "--learningRate", dest="learningRate", default="0.01")
    parser.add_option("-k", "--learningrateDecay", dest="learningrateDecay", default="0.0002")
    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="birnn", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    # Parameters
    batch_size = int(options.batchSize)
    learning_rate = float(options.learningRate)
    learning_rate_decay = float(options.learningrateDecay)
    if options.action == "train":
        def get_end_trigger():
            if options.endTriggerType.lower() == "epoch":
                return MaxEpoch(options.endTriggerNum)
            else:
                return MaxIteration(options.endTriggerNum)

        train_data = get_mnist(sc, "train", options.dataPath)\
             .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),rec_tuple[1])).map(lambda t: Sample.from_ndarray(t[0], t[1]))
        test_data = get_mnist(sc, "test", options.dataPath)\
        .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD), rec_tuple[1])).map(lambda t: Sample.from_ndarray(t[0], t[1]))
        
       
        
        optimizer = Optimizer(
            model=build_model(n_input, n_hidden, n_classes),
            training_rdd=train_data,
            criterion=CrossEntropyCriterion(),
            optim_method=SGD(learningrate=learning_rate, learningrate_decay=learning_rate_decay),
            end_trigger=get_end_trigger(),
            batch_size=options.batchSize)
        optimizer.set_validation(
            batch_size=options.batchSize,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=[Top1Accuracy()]
        )
        optimizer.set_checkpoint(EveryEpoch(), options.checkpointPath)
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        #test_data = get_mnist(sc, "test").map(normalizer(mnist.TEST_MEAN, mnist.TEST_STD))
        test_data = get_mnist(sc, "test", options.dataPath) \
            .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD),
                                    rec_tuple[1])) \
            .map(lambda t: Sample.from_ndarray(t[0], t[1]))
        model = Model.load(options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    sc.stop()

