import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

image_size = 28
num_labels = 10
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


# train_labels = tf.one_hot(tf.cast(train_labels, tf.int32), 10, 1, 0)
# valid_labels = tf.one_hot(tf.cast(valid_labels, tf.int32), 10, 1, 0)
# test_labels = tf.one_hot(tf.cast(test_labels, tf.int32), 10, 1, 0)
# train_labels = tf.cast(train_labels, tf.int32)
# valid_labels = tf.cast(valid_labels, tf.int32)
# test_labels = tf.cast(test_labels, tf.int32)

# train_dataset = tf.reshape(train_dataset, [-1, 784])
# valid_dataset = tf.reshape(valid_dataset, [-1, 784])
# test_dataset = tf.reshape(test_dataset, [-1, 784])

train_dataset = train_dataset.reshape(-1, 784)
valid_dataset = valid_dataset.reshape(-1, 784)
test_dataset = test_dataset.reshape(-1, 784)

def input_fn():
    return tf.cast(train_dataset, tf.float32), tf.cast(train_labels, tf.int32)

def test_input_fn():
    return tf.cast(test_dataset, tf.float32), tf.cast(test_labels, tf.int32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1024, 10],
    n_classes=10,
    model_dir="./logs",
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ))

classifier.fit(input_fn=input_fn, steps=2000)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)['accuracy']
print("Accuracy: {0:f}".format(accuracy_score))
