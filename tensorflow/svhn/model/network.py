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
        self.softmax_config = []
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale
        self.softmax_weights = []
        self.softmax_biases = []

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

    def add_softmax_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        self.softmax_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes], name=name + '_biases'))
            self.softmax_weights.append(weights)
            self.softmax_biases.append(biases)
            self.train_summaries.append(tf.histogram_summary(str(len(self.softmax_weights)) + '_softmax_weights', weights))
            self.train_summaries.append(tf.histogram_summary(str(len(self.softmax_biases)) + '_softmax_biases', biases))


    def apply_regularization(self, _lambda):
        '''
        L2 regularization for the fully connected parameters
        '''
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        regularization += tf.nn.l2_loss(self.softmax_weights[0]) + tf.nn.l2_loss(self.softmax_biases[0])
        return _lambda * regularization

    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_sample_shape):
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            # self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_train_labels_length = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_length')
            self.tf_train_labels_1 = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_1')
            self.tf_train_labels_2 = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_2')
            self.tf_train_labels_3 = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_3')
            self.tf_train_labels_4 = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_4')
            self.tf_train_labels_5 = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels_5')
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

            # Define softmax layers
            # original_data_flow = data_flow
            # result_array = None
            # self.loss = 0
            # for i, (weights, biases, config) in enumerate(zip(self.softmax_weights, self.softmax_biases, self.softmax_config)):
            #     with tf.name_scope(config['name'] + 'model'):
            #         data_flow = tf.matmul(original_data_flow, weights) + biases
            #         if config['activation'] == 'relu':
            #             data_flow = tf.nn.relu(data_flow)
            #         elif config['activation'] == 'softmax':
            #             data_flow = tf.nn.softmax(data_flow)
            #         elif config['activation'] is None:
            #             pass
            #         else:
            #             raise Exception('Activation function can only be Relu or None right now. You passed', config['activation'])
            #     if i == 0:
            #         result_array = data_flow
            #     else:
            #         result_array = tf.concat(1, [result_array, data_flow])
            #
            #     if train:
            #         with tf.name_scope('loss'):
            #             print(data_flow.get_shape)
            #             print(self.tf_train_labels[:,i].get_shape)
            #             self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(data_flow, self.tf_train_labels[:, i]))
            # return result_array

            # Define softmax layers
            # data_flow = tf.matmul(data_flow, self.softmax_weights[0]) + self.softmax_biases[0]
            # return data_flow

            data_flow_length = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[0]) + self.softmax_biases[0])
            data_flow_1 = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[1]) + self.softmax_biases[1])
            data_flow_2 = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[2]) + self.softmax_biases[2])
            data_flow_3 = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[3]) + self.softmax_biases[3])
            data_flow_4 = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[4]) + self.softmax_biases[4])
            data_flow_5 = tf.nn.softmax(tf.matmul(data_flow, self.softmax_weights[5]) + self.softmax_biases[5])

            return data_flow_length, data_flow_1, data_flow_2, data_flow_3, data_flow_4, data_flow_5

        # Training computation
        # print('Start use mode with train samples')
        self.labels_length, self.label1, self.label2, self.label3, self.label4, self.label5 = model(self.tf_train_samples)
        # logits = model(self.tf_train_samples)

        with tf.name_scope('loss'):
            # print('define loss')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.labels_length, self.tf_train_labels_length))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.label1, self.tf_train_labels_1))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.label2, self.tf_train_labels_2))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.label3, self.tf_train_labels_3))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.label4, self.tf_train_labels_4))
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.label5, self.tf_train_labels_5))

            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels_length))
            # self.loss += self.apply_regularization(_lambda=5e-4)
            # self.train_summaries.append(tf.scalar_summary('Loss', self.loss))

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
        # with tf.name_scope('train'):
            # print('define train model')
            # self.train_prediction = tf.nn.softmax(logits, name='train_prediction')

        with tf.name_scope('test'):
            print('define test model')
            self.test_labels_length, self.test_label1, self.test_label2, \
            self.test_label3, self.test_label4, self.test_label5 = model(self.tf_test_samples, train=False)
            # self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')


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
            # Train Set (33401, 64, 64, 1) (33401, 6)
            # Test Set(13068, 64, 64, 1) (13068, 6)

            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps, chunkSize=self.train_batch_size):
                # print(samples.shape, self.tf_train_samples.get_shape)
                # print(labels[:, 0].shape, self.tf_train_labels_length.get_shape)
                # print(labels[:, 1].shape, self.tf_train_labels_1.get_shape)
                # print(labels[:, 2].shape, self.tf_train_labels_2.get_shape)
                # print(labels[:, 3].shape, self.tf_train_labels_3.get_shape)
                # print(labels[:, 4].shape, self.tf_train_labels_4.get_shape)
                # print(labels[:, 5].shape, self.tf_train_labels_5.get_shape)

                _, l, \
                labels_length, label1, label2, label3, label4, label5,\
                = session.run(
                    [self.optimizer, self.loss,
                     self.labels_length, self.label1, self.label2, self.label3, self.label4, self.label5,],
                    feed_dict={self.tf_train_samples: samples,
                               self.tf_train_labels_length: labels[:, 0],
                               self.tf_train_labels_1: labels[:, 1],
                               self.tf_train_labels_2: labels[:, 2],
                               self.tf_train_labels_3: labels[:, 3],
                               self.tf_train_labels_4: labels[:, 4],
                               self.tf_train_labels_5: labels[:, 5]
                               }
                )

                # _, l, predictions, summary = session.run(
                #     [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                #     feed_dict={self.tf_train_samples: samples, self.tf_train_labels_length: labels[:, 1]}
                # )

                # self.writer.add_summary(summary, i)
                # print(np.array(predictions).transpose((1,0,2)).shape, labels.shape)
                # print(np.array(predictions)[0,0], labels[0,0])
                # predictions = np.array(predictions).transpose((1,0,2))
                # print(predictions[0,0], labels[0,0])
                # accuracy, _ = self.accuracy(predictions, labels)
                # print(type(labels_length), labels_length.shape)

                predictions = np.hstack((labels_length, label1, label2, label3, label4, label5))
                predictions = predictions.reshape((64, 6, 10))
                accuracy, _ = self.accuracy(predictions, labels)
                # accuracy, _ = self.accuracy2(predictions, labels[:, 1])



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

    def accuracy2(self, predictions, labels, need_confusion_matrix=False):
        '''
        Calculate the accuracy and confusion matrix
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        print(_predictions[:10], _labels[:10])
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        '''
        Calculate the accuracy and confusion matrix
        '''
        # predictions 64x6x10
        #
        # _predictions = np.argmax(predictions, 2) # 64 x 6
        # print("predictions and labels: ")
        # print(predictions[0])
        _predictions = self.getLabel(predictions)
        # print(_predictions[0])
        # _labels = np.argmax(labels, 2)
        _labels = self.getLabel(labels)
        print(_predictions[:10], _labels[:10])

        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

    def getLabel(self, labels):
        # label n x 6 x 10
        label_array = np.argmax(labels, 2) # n x 6
        # print('first prediction label: ', label_array)
        label_output = []
        for label in label_array:
            length = label[0]
            # print("length: ", length)
            label_str = str(label[1])
            for i in range(2, len(label)):
                if len(label_str) < length:
                    label_str += str(label[i])
            # print("label str is {0} and the length is {1}".format(label_str, length))
            label_output.append(int(label_str))
        return label_output
    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        # print('tensor shape: ', tensor.get_shape)
        filter_map = tensor[-1]
        # print('filter_map shape1: ', filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        # print('filter_map shape2: ', filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        # print("how many:", how_many)
        self.test_summaries.append(tf.image_summary(name, tensor=filter_map, max_images=how_many))
