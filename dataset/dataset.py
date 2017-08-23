import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy
import matplotlib.pyplot as plt

def elastic_transform(image, alpha=34, sigma=8, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
 
 
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = numpy.random.RandomState(None)
 
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
 
    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
 
    new_image = map_coordinates(image, indices, order=1).reshape(shape)
    return new_image

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
    def __init__(self, inputs, labels, expand=False, random = False):
        self.__inputs = np.array(inputs)
        self.__labels = np.array(labels)
        self.__index_in_epoch = 0
        self.__epochs_completed = 0

        expanded_examples = []
        expanded_labels = []
        tot = len(self.__inputs)
        i = 0
        if(expand):
            for x,y in zip(self.__inputs, self.__labels):
                print i / float(tot)
                i = i + 1
                bg_value = np.median(x)
                shape = x.shape
                image = x.reshape(28,28)
                for _ in range(8):
                    angle = np.random.randint(-10,10,1)
                    new_image = ndimage.rotate(image, angle, reshape=False, cval=bg_value)
                    shift = np.random.randint(-2,2,2)
                    new_image_ = ndimage.shift(new_image, shift, cval=bg_value)
                    new_image___ = elastic_transform(new_image_)
                    expanded_examples.append(new_image___.reshape(shape))
                    expanded_labels.append(y)
            self.__inputs = np.concatenate((self.__inputs, expanded_examples), axis=0)
            self.__labels = np.concatenate((self.__labels, expanded_labels), axis=0)
        print self.__labels.size
        if self.__labels.size > 1 and random == True:
            res = zip(self.__inputs, self.__labels)
            np.random.shuffle(res)
            self.__inputs, self.__labels = zip(*res)
            self.__inputs = np.array(self.__inputs)
            self.__labels = np.array(self.__labels)
        elif random == True:
            np.random.shuffle(self.__inputs)

    def get_all(self):
        return self.__inputs, self.__labels
    

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

            #res = zip(self.__inputs, self.__labels)
            #np.random.shuffle(res)
            #self.__inputs, self.__labels = zip(*res)
            #self.__inputs = np.array(self.__inputs)
            #self.__labels = np.array(self.__labels)
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
    def __init__(self, test_filename, train_filename, generate_test_set=False):
        train_inputs, train_labels = dataset_from_csv(train_filename, train=True)

        if(generate_test_set == True):
            self.train = Dataset(train_inputs[:len(train_inputs)*3/4]/255, train_labels[:len(train_inputs)*3/4], expand=True, random=True)
            self.test =  Dataset(train_inputs[len(train_inputs)*3/4+1:],   train_labels[len(train_inputs)*3/4+1:])
        else:
            print('quux')
            self.train = Dataset(train_inputs/255., train_labels, expand=True, random=True)
            test_inputs, test_labels = dataset_from_csv(test_filename)
            self.test = Dataset(test_inputs/255., test_labels)
            print self.test.inputs.size

    def write_results(self, dataset, name='results.csv'):
        results = pd.DataFrame({'ImageId': pd.Series(range(1, len(dataset.labels)+1)), 'Label': pd.Series(dataset.labels)})
        results.to_csv(name, index=False)
