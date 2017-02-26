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


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)

        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations


def define_input(image_size=28, number_labels=10, dropout_rate=0.9):
    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32, [None, image_size * image_size], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, number_labels], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, image_size, image_size, 1])
        tf.summary.image('input', image_shaped_input, number_labels)
    return x, y_


def main(learning_rate=0.05, max_steps=3001, batch_size=128):
    sess = tf.InteractiveSession()

    x, y_ = define_input(28, 10)
    hidden1 = nn_layer(x, 784, 1024, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        droped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(hidden1, 1024, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./summary/train', sess.graph)

    tf.global_variables_initializer().run()

    for step in range(max_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]
        feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 0.99}

        if step % 500 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _, acc = sess.run([merged, train_step, accuracy],
                                       feed_dict=feed_dict,
                                       options=run_options,
                                       run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)
            print('Adding run metadata for %s and the accuracy is %s' %
                  (step, acc))
        else:
            summary, _, acc = sess.run(
                [merged, train_step, accuracy], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

        if (step % 500 == 0):
            summary, acc = sess.run(
                [merged, accuracy], feed_dict={x: valid_dataset, y_: valid_labels, keep_prob: 1})

            train_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))

    summary, acc = sess.run(
        [merged, accuracy], feed_dict={x: test_dataset, y_: test_labels, keep_prob: 1})

    train_writer.add_summary(summary, step + 1)
    print('Total Accuracy at step %s: %s' % (step + 1, acc))

    train_writer.close()

main(0.02, 5001, 128)
