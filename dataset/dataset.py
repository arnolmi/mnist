import pandas as pd
import numpy as np
import tensorflow as tf
class BatchSizeOutOfBoundsException(Exception):
    def __init__(self, *dErrorArguments):
        Exception.__init__(self, "Batch Size was larger than dataset size {}"
                           .format(dErrorArguments))


class DataSetSizeException(Exception):
    def __init__(self):
        Exception.__init__(self, "DataSet Size was smaller than window size")


def dataset_from_csv(filename, train = False):
    raw_data = pd.read_csv(filename)
    if(train):
        data = raw_data.drop(['label'], axis=1)
        data = data.values.astype(dtype=np.float32)
        data = data.reshape(len(data), 28*28)
        labels = raw_data['label'].tolist()
    
        labels = tf.one_hot(labels, depth=10)
        labels = tf.Session().run(labels).astype(dtype=np.float64)

    else:
        data = raw_data.values.astype(dtype=np.float32)
        labels = None
    return data, labels


class Dataset(object):
    def __init__(self, inputs, labels):
        self.__inputs = np.array(inputs)
        self.__labels = np.array(labels)
        self.__index_in_epoch = 0
        self.__epochs_completed = 0

    def next_batch(self, batch_size):
        if batch_size > self.__inputs.shape[0]:
            raise BatchSizeOutOfBoundsException(batch_size, self.__inputs.shape[0])

        start = self.__index_in_epoch
        first_part_x = np.array([])
        first_part_y = np.array([])

        if self.__index_in_epoch + batch_size > self.__inputs.shape[0]:
            # Grab the first part
            first_part_x = self.__inputs[start:self.__inputs.shape[0]]
            first_part_y = self.__labels[start:self.__labels.shape[0]]
            batch_size -= first_part_x.shape[0]

            res = zip(self.__inputs, self.__labels)
            np.random.shuffle(res)
            self.__inputs, self.__labels = zip(*res)
            self.__inputs = np.array(self.__inputs)
            self.__labels = np.array(self.__labels)
            start = 0
            self.__index_in_epoch = 0
            self.__epochs_completed += 1
            end = start + batch_size
            x_res = np.concatenate((first_part_x, self.__inputs[start:end]))
            y_res = np.concatenate((first_part_y, self.__labels[start:end]))

            self.__index_in_epoch += batch_size
            return x_res, y_res
        else:
            end = start + batch_size
            self.__index_in_epoch += batch_size
            return self.__inputs[start:end], self.__labels[start:end]

    @property
    def inputs(self):
        return self.__inputs

    @property
    def labels(self):
        return self.__labels

    @property
    def epoch(self):
        return self.__epochs_completed


class MnistDataset(object):
    def __init__(self, test_filename, train_filename):
        train_inputs, train_labels = dataset_from_csv(train_filename, train=True)
        self.train = Dataset(train_inputs/255., train_labels)

        test_inputs, test_labels = dataset_from_csv(test_filename)
        self.test = Dataset(test_inputs/255., test_labels)

    def write_results(self, dataset, name='results.csv'):
        results = pd.DataFrame({'ImageId': pd.Series(range(1, len(dataset.labels)+1)), 'Label': pd.Series(dataset.labels)})
        results.to_csv(name, index=False)
