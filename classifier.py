# -*- coding: utf-8 -*-

import csv
import keras

from numpy import array as array

def load_dataset(path):
    with open(path, "r") as dataset:
        dataset_reader = csv.reader(dataset)

        next(dataset_reader) #Skip headers
        dataset_list = [row for row in dataset_reader]

    data_cut1 = int(len(dataset_list) * 0.7)
    data_cut2 = int(data_cut1 + len(dataset_list) * 0.15)

    train_data = array(dataset_list[:data_cut1])
    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1]

    validation_data = array(dataset_list[data_cut1:data_cut2])
    validation_data_x = validation_data[:, :-1]
    validation_data_y = validation_data[:, -1]

    test_data = array(dataset_list[data_cut2:])
    test_data_x = test_data[:, :-1]
    test_data_y = test_data[:, -1]

    return ((train_data_x, train_data_y), \
            (validation_data_x, validation_data_y), \
            (test_data_x, test_data_y))

def main():
    #Male is 0, Female is 1
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset("data/voice.csv")

    batch_size = 150
    num_classes = 2
    epochs = 20

    train_y = keras.utils.to_categorical(train_y, num_classes)
    valid_y = keras.utils.to_categorical(valid_y, num_classes)
    test_y = keras.utils.to_categorical(test_y, num_classes)



if __name__ == "__main__":
    main()
