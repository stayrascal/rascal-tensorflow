import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np


class Network(object):
    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate, optimize_method='adam', save_path='model/default.ckpt'):
        '''
        :param train_batch_size:
        :param test_batch_size:
        :param pooling_scale:
        '''
        self.optimize_method = optimize_method
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size

        # Hyper Parameters
        self.conv_config = []
        self.fc_config = []
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale

        # Graph Related
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None

        # statistics
        self.merged = None
        self.writer = None
        self.train_summaries = []
        self.test_summaries = []

        # save train model
        self.saver = None
        self.save_path = save_path

    def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
        '''
        This function doesn't define operations in the graph, but only store config in self.conv_config
        '''
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_depth], name=name + '_biases'))
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        '''
        Add fc layer config to self.fc_config
        '''
        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes], name=name + '_biases'))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.histogram_summary(str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.histogram_summary(str(len(self.fc_biases)) + '_biases', biases))

    def apply_regularization(self, _lambda):
        '''
        L2 regularization for the fully connected parameters
        '''
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        return _lambda * regularization

    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_sample_shape):
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_sample_shape, name='tf_test_samples')

    def define_model(self):
        def model(data_flow, train=True):
            # Define Convolution layers
            for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                with tf.name_scope(config['name'] + 'model'):
                    with tf.name_scope('convolution'):
                        # Default 1,1,1,1 stride and SAME padding
                        data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                        data_flow = data_flow + biases
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1),
                                                      name=config['name'] + '_conv')
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1),
                                                      name=config['name'] + '_relu')
                    else:
                        raise Exception('Activation function can only be Relu right now. You passed', config['activation'])

                    if config['pooling']:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                            strides=[1, self.pooling_scale, self.pooling_scale, 1],
                            padding='SAME'
                        )
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1) // 2,
                                                      name=config['name'] + '_pooling')

            # Define fully connected layers
            for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                if i == 0:
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
                with tf.name_scope(config['name'] + 'model'):
                    if train and i == len(self.fc_weights) - 1:
                        data_flow = tf.nn.dropout(data_flow, self.dropout_rate, seed=4926)
                    data_flow = tf.matmul(data_flow, weights) + biases
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                    elif config['activation'] is None:
                        pass
                    else:
                        raise Exception('Activation function can only be Relu or None right now. You passed', config['activation'])
            return data_flow

        # Training computation
        # print('Start use mode with train samples')
        logits = model(self.tf_train_samples)
        with tf.name_scope('loss'):
            # print('define loss')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
            self.loss += self.apply_regularization(_lambda=5e-4)
            self.train_summaries.append(tf.scalar_summary('Loss', self.loss))

        # Optimizer
        with tf.name_scope('optimizer'):
            # learning rate decay
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.base_learning_rate,
                global_step=global_step * self.train_batch_size,
                decay_steps=100,
                decay_rate=self.decay_rate,
                staircase=True
            )

            # print('define optimizer')
            if (self.optimize_method == 'gradient'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            elif (self.optimize_method == 'momentum'):
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(self.loss)
            elif (self.optimize_method == 'adam'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train'):
            # print('define train model')
            self.train_prediction = tf.nn.softmax(logits, name='train_prediction')

        with tf.name_scope('test'):
            # print('define test model')
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')

        self.merged_train_summary = tf.merge_summary(self.train_summaries)
        self.merged_test_summary = tf.merge_summary(self.test_summaries)

        self.saver = tf.train.Saver(tf.all_variables())

    def run(self, train_samples, train_labels, test_samples, test_labels, *, train_data_iterator, iteration_steps, test_data_iterator):
        '''
        Using Session
        :data_iterator: a function that yields chuck of data
        '''
        self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())

        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            # Train
            print('Start Training')
            # batch 1000
            for i, samples, labels in train_data_iterator(train_samples, train_labels, interation_steps=iteration_steps,
                                                          chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )
                self.writer.add_summary(summary, i)

                # Labels is True labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)

            # Testing
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))
            # print_confusion_matrix(np.add.reduce(confusionMatrices))

    def train(self, train_samples, train_labels, *, data_iterator, iteration_steps):
        self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            print('Start Training')
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps, chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )

                self.writer.add_summary(summary, i)
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)

            import os
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.mkdir(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, *, data_iterator):
        if self.saver is None:
            self.define_model()
        if self.writer is None:
            self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)
            accuracies = []
            confusion_matrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusion_matrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusion_matrices))

    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion Matrix:', len(confusionMatrix))
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column) * (np.sum(column) / 26000))
            print(column[i] / np.sum(column), )
        print('\n', np.sum(confusionMatrix), a)

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        '''
        Calculate the accuracy and confusion matrix
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        # print('tensor shape: ', tensor.get_shape)
        filter_map = tensor[-1]
        # print('filter_map shape1: ', filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        # print('filter_map shape2: ', filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        # print("how many:", how_many)
        self.test_summaries.append(tf.image_summary(name, tensor=filter_map, max_images=how_many))
