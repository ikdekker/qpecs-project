import pandas as pd


def recommend(model_selected):
    df = pd.read_csv("data_m.csv")
    df = df.sort_values(by=['accuracy', 'responsetime'], ascending=(False, False))

    row = df[df.model == model_selected].iloc[0]
    # Get array of system-, and hyper-parameters that gives the
    # best performance In other words, recommend this to the user
    model = row.values[0]
    cores = row.values[1]
    ram = row.values[2]
    batch_size = row.values[3]
    learning_rate = row.values[4]
    learning_rate_decay = row.values[5]

    print(
        '{}-model {}-Cores {}-GBRAM {}-batchSize {}-learningRate {}-learningRateDecay'
        .format(
            model, cores, ram, batch_size, learning_rate, learning_rate_decay
        )
    )

    return cores, ram, batch_size, learning_rate, learning_rate_decay, model