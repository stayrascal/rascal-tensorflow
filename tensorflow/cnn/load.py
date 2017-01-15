from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    '''
    (height, width, channels, images) => (images, height, width, channels)
    '''
    samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float)
    labels = (np.arange(10) + 1 == labels[:, None]).astype(np.float)
    labels = labels.reshape(labels.shape[0], labels.shape[1] * labels.shape[2])
    return samples, labels


def normalize(samples):
    '''
    samples: (images, height, width, channels)
    '''
    pixel_sum = np.add.reduce(samples, keepdims=True, axis=3)
    pixel_average = pixel_sum / 3.0  # save memory and speed up training
    return pixel_average / 128.0 - 1.0


def distribution(labels, name):
    '''
     show distribution of labels
    '''
    x = np.unique(labels)
    y = np.bincount(labels.ravel())[1:]
    y_pos = np.arange(len(x)) + 1
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    '''
    Show picture
    '''
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()

num_labels = 10
image_size = 32
num_channels = 1

# train = load("../data/train_32x32.mat")
test = load("../data/test_32x32.mat")
extra = load("../data/extra_32x32.mat")

train_samples = extra['X']
train_labels = extra['y']
test_samples = test['X']
test_labels = test['y']
# valid_samples = train['X']
# valid_labels = train['y']

n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)
# n_valid_samples, _valid_labels = reformat(valid_samples, valid_labels)

# n_train_samples = n_train_samples[:5]
# _train_labels = _train_labels[:5]

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)
# _valid_samples = normalize(n_valid_samples)


if __name__ == '__main__':
    pass
    # save_data('train.pickle',
    #           'train_dataset', _train_samples,
    #           'train_label', _train_labels)
    # print(train_samples.shape)
    # print(extra['X'].shape)
    # print(_test_samples.shape)
    # print(_train_labels.shape)
    # for i in range(10):
    #     inspect(_test_samples, _test_labels, 4)
    # distribution(train_labels, 'Train Labels')
