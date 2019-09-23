import neural_network as net
import numpy as np
import pandas as pd
import utils as u
import data_bird as db

def make_kfold_and_save(pandas_data, filename_train, filename_validation):
    k = 10
    fold_size = len(pandas_data) / k
    data = [(pandas_data.iloc[i][1:], u.one_hot(10, pandas_data.iloc[i][0])) 
        for i in range(0, len(pandas_data))]

    train_data = data[:int(fold_size*(k-1))]
    validation_data = data[int(fold_size*(k-1)):]

    # db.save_model(train_data, filename_train)
    # db.save_model(validation_data, filename_validation)
    return train_data, validation_data

if __name__ == "__main__":
    training_data_filename = "data/train_data.pickle"
    validation_data_filename = "data/validation_data.pickle"

    data = pd.read_csv("C:\\Users\\patyc\\Documents\\GitHub\\ml-studies\\ann\\data\\train.csv")
    training_data, validation_data = make_kfold_and_save(data, training_data_filename, validation_data_filename)
    
    # training_data = db.load_model(training_data_filename)
    # validation_data = db.load_model(validation_data_filename)

    nn = net.NeuralNetwork([28*28, 32, 10])

    # nn.train(training_data)
    # db.save_model(nn.serialize(), "ann_mnist.pickle")

    nn.deserialize(db.load_model("ann_mnist.pickle"))

    correct = 0
    for input, label in validation_data:
        output = nn.predict(input)[-1]
        if np.argmax(output) == np.argmax(label):
            correct += 1
    print("accuracy: ", correct / len(validation_data))