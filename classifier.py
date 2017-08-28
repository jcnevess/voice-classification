# -*- coding: utf-8 -*-

import csv
import keras
import random

from numpy import array as array
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from keras import optimizers


def load_dataset(path):
    with open(path, "r") as dataset:
        dataset_reader = csv.reader(dataset)

        next(dataset_reader)  # Skip headers
        dataset_list = [row for row in dataset_reader]

    male_data = dataset_list[:len(dataset_list) // 2]
    female_data = dataset_list[len(dataset_list) // 2:]

    data_cut1 = int(len(male_data) * 0.7)
    data_cut2 = int(data_cut1 + len(male_data) * 0.15)

    train_list = male_data[:data_cut1] + female_data[:data_cut1]
    random.shuffle(train_list)
    train_data = array(train_list)
    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1]

    validation_list = male_data[data_cut1:data_cut2] + female_data[data_cut1:data_cut2]
    random.shuffle(validation_list)
    validation_data = array(validation_list)
    validation_data_x = validation_data[:, :-1]
    validation_data_y = validation_data[:, -1]

    test_list = male_data[data_cut2:] + female_data[data_cut2:]
    random.shuffle(test_list)
    test_data = array(test_list)
    test_data_x = test_data[:, :-1]
    test_data_y = test_data[:, -1]

    return ((train_data_x, train_data_y),
            (validation_data_x, validation_data_y),
            (test_data_x, test_data_y))


def create_network(num_features, layer_sizes, activation):
    # define the layers
    network = Sequential()
    network.add(Dense(layer_sizes[0], input_dim=num_features))

    for layer_size in layer_sizes[1:]:
        network.add(Dense(layer_size, activation=activation))

    return network


def run_network():
    # Male is 0, Female is 1
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset("data/voice.csv")

    batch_size = 50
    num_classes = 2
    epochs = 50

    train_y = keras.utils.to_categorical(train_y, num_classes)
    valid_y = keras.utils.to_categorical(valid_y, num_classes)
    test_y = keras.utils.to_categorical(test_y, num_classes)

    model = create_network(
                num_features=20,
                layer_sizes=[5, 2],  # [10, 2] [5, 2] [15, 7, 2] [10, 5, 2] [15, 10, 5, 2]
                activation="sigmoid")

    # configure the learning process
    opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # train the model
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(valid_x, valid_y))

    score = model.evaluate(test_x, test_y,
                           batch_size=batch_size)

    return score[0]  # accuracy


def main():
    iterations = 30
    result = 0

    for i in range(iterations):
        print("\n\n\nRunning iteration %d\n\n" % (i + 1))
        result += run_network()

    result = result / iterations
    print("\n\nMean accuracy after %d iterations: %s" % (iterations, result))

    backend.clear_session()


if __name__ == "__main__":
    main()
