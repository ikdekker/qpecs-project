import pandas as pd
from sklearn import preprocessing, tree


def recommend(df, model_selected):
    df = df.sort_values(
        by=['accuracy', 'responsetime'], ascending=(False, False)
    )

    row = df[df.model == model_selected].iloc[0] # The best we found for this model

    # Get array of system-, and hyper-parameters that gives the
    # best performance. In other words, recommend this to the user
    model = row.values[0]
    cores = row.values[1]
    ram = row.values[2]
    batch_size = row.values[3]
    learning_rate = row.values[4]
    learning_rate_decay = row.values[5]

    print(
        'Better: {}-model {}-Cores {}-GBRAM {}-batchSize {}-learningRate {}-learningRateDecay'
        .format(
            model, cores, ram, batch_size, learning_rate, learning_rate_decay
        )
    )

    return cores, ram, batch_size, learning_rate, learning_rate_decay, model


if __name__ == "__main__":
    ############################
    # Simulate a job coming in #
    ############################
    request = {
        'model': ['lenet5'],
        'Cores': [1],  # 1 or 4
        'GBRAM': [1],  # 1 or 8
        'batchSize': [64],  # 64 or 512
        'learningRate': [0.001],  # 0.1 or 0.001
        'learningRateDecay': [0.001],  # 0.001 or 0.0001
        'status': ['slow']  # This is not needed but for removing the error
    }

    job = pd.DataFrame.from_dict(request)

    ##################################################################################
    # load the dataset, append incoming job, train and classify if job can be better #
    ##################################################################################
    df = pd.read_csv("data_m.csv")
    df_train = df
    df_train = df_train.append(job, sort=False)

    df_train = df_train.apply(preprocessing.LabelEncoder().fit_transform)
    df_train = df_train[
        [
            'model', 'Cores', 'GBRAM', 'batchSize', 'learningRate', 
            'learningRateDecay', 'accuracy', 'responsetime', 'status'
        ]
    ]
    df_train = df_train[:-1] # Since the incoming job is now encoded, we can remove it

    train_x = df_train.iloc[:,0:6] # features (ignore accuracy and responsetime)
    train_y = df_train.iloc[:,8]  # target_dict(request)

    # # Train a DT model on the original features
    dt = tree.DecisionTreeClassifier(max_depth=5).fit(train_x, train_y)

    # Compute the predicted labels on test data
    predicted_output = dt.predict(train_x.iloc[[-1], [0, 1, 2, 3, 4, 5]])[0]
    if predicted_output: # 1, means it's slow and needs fixing
        recommend(df, request['model'][0])
    else: # Already good, not need to fix it
        pass