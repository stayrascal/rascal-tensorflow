import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


class Network(object):
    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate, optimize_method='adam', save_path='model/default.ckpt'):
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
        self.tf_validation_samples = None
        self.tf_validation_labels = None

        # statistics
        self.merged = None
        self.writer = None
        self.train_summaries = []
        self.test_summaries = []

        # save train model
        self.saver = None
        self.save_path = save_path

    def init_weights_conv(self, shape, name):
        # 变量作用域机制
        # 方法 tf.get_variable() 用来获取或创建一个变量，而不是直接调用tf.Variable.它采用的不是像`tf.Variable这样直接获取值来初始化的方法.
        return tf.get_variable(shape=shape, name=name, initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def init_weights_fc(self, shape, name):
        return tf.get_variable(shape=shape, name=name, initializer=tf.contrib.layers.xavier_initializer())

    def init_biases(self, shape, name):
        return tf.Variable(tf.constant(1.0, shape=shape), name=name)

    def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        with tf.name_scope(name):
            weights = self.init_weights_conv(shape=[patch_size, patch_size, in_depth, out_depth], name=name + '_weights')
            biases = self.init_biases(shape=[out_depth], name=name + '_biases')
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)
            self.train_summaries.append(tf.histogram_summary(str(len(self.conv_weights)) + '_conv_weights', weights))
            self.train_summaries.append(tf.histogram_summary(str(len(self.conv_biases)) + '_conv_biases', biases))

    def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = self.init_weights_fc(shape=[in_num_nodes, out_num_nodes], name=name + '_weights')
            biases = self.init_biases(shape=[out_num_nodes], name=name + '_biases')
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.histogram_summary(str(len(self.fc_weights)) + '_fc_weights', weights))
            self.train_summaries.append(tf.histogram_summary(str(len(self.fc_biases)) + '_fc_biases', biases))

    def add_softmax_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        self.softmax_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = self.init_weights_fc(shape=[in_num_nodes, out_num_nodes], name=name + '_weights')
            biases = self.init_biases(shape=[out_num_nodes], name=name + '_biases')
            self.softmax_weights.append(weights)
            self.softmax_biases.append(biases)
            self.train_summaries.append(tf.histogram_summary(str(len(self.softmax_weights)) + '_softmax_weights', weights))
            self.train_summaries.append(tf.histogram_summary(str(len(self.softmax_biases)) + '_softmax_biases', biases))

    def apply_regularization(self, _lambda):
        '''
        L2 regularization for the fully and softmax connected parameters
        '''
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        for weights, biases in zip(self.softmax_weights, self.softmax_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        return _lambda * regularization

    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_samples_shape):
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.int32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def model(self, data_flow, train=True):
        # Define Convolution layers
        self.image_size = 32
        for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
            with tf.name_scope(config['name'] + 'model'):
                with tf.name_scope('convolution'):
                    # Default 1,1,1,1 stride and VALID padding
                    data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
                    data_flow = data_flow + biases
                    if not train:
                        self.image_size = self.image_size - config['patch_size'] + 1
                        self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=self.image_size,
                                                  name=config['name'] + '_conv')
                if config['activation'] == 'relu':
                    data_flow = tf.nn.relu(data_flow)
                    if not train:
                        self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=self.image_size,
                                                  name=config['name'] + '_relu')
                else:
                    raise Exception('Activation function can only be Relu right now. You passed', config['activation'])

                if config['pooling']:
                    data_flow = tf.nn.max_pool(
                        data_flow,
                        ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                        strides=[1, self.pooling_scale, self.pooling_scale, 1],
                        padding='VALID'
                    )
                    if not train:
                        self.image_size = self.image_size // 2
                        self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=self.image_size,
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

        # Define softmax connected layers
        original_data_flow = data_flow
        logits = []
        for i, (weights, biases, config) in enumerate(zip(self.softmax_weights, self.softmax_biases, self.softmax_config)):
            with tf.name_scope(config['name'] + 'model'):
                data_flow = tf.matmul(original_data_flow, weights) + biases
                logits.append(data_flow)
        return logits

    def softmax_combine(self, dataset, train):
        dataset = self.model(dataset, train)
        return tf.pack([
            tf.nn.softmax(dataset[0]),
            tf.nn.softmax(dataset[1]),
            tf.nn.softmax(dataset[2]),
            tf.nn.softmax(dataset[3]),
            tf.nn.softmax(dataset[4]),
            tf.nn.softmax(dataset[5])])

    def define_model(self):
        # Training computation
        print('Start use mode with train samples')
        logits = self.model(self.tf_train_samples)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], self.tf_train_labels[:, 0]))
            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], self.tf_train_labels[:, 1]))
            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], self.tf_train_labels[:, 2]))
            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], self.tf_train_labels[:, 3]))
            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], self.tf_train_labels[:, 4]))
            self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[5], self.tf_train_labels[:, 5]), name="loss")
            # self.loss += self.apply_regularization(_lambda=5e-4)
            # Add scalar summary for cost
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
                staircase=True,
                name='learning_rate'
            )
            if (self.optimize_method == 'gradient'):
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            elif (self.optimize_method == 'momentum'):
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(self.loss)
            elif (self.optimize_method == 'adam'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Predictions for the training, and test data.
        with tf.name_scope('train'):
            self.train_prediction = tf.pack(logits)
        with tf.name_scope('test'):
            self.test_prediction = self.softmax_combine(self.tf_test_samples, train=False)

        self.merged_train_summary = tf.merge_summary(self.train_summaries)
        self.merged_test_summary = tf.merge_summary(self.test_summaries)

        self.saver = tf.train.Saver(tf.all_variables())

    def run(self, train_samples, train_labels, test_samples, test_labels, *, train_data_iterator, iteration_steps, test_data_iterator):
        self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            print('Start Training')
            for i, samples, labels in train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                          chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels})
                self.writer.add_summary(summary, i)
                accuracy = self.accuracy(predictions, labels)

                if i % 500 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)

            # Testing
            accuracies = []
            for i, samples, labels in test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy = self.accuracy(result, labels)
                # print('the predict is:\n {} \nand the label is:\n {}'.format(self.get_prediction(result[1:6]), self.get_label(labels[:, 1:6])))
                accuracies.append(accuracy)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))

    def train(self, train_samples, train_labels, *, data_iterator, iteration_steps):
        self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            print('Start Training')
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps, chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels})
                self.writer.add_summary(summary, i)
                accuracy = self.accuracy(predictions, labels)

                if i % 500 == 0:
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

            print('Start Testing')

            accuracies = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy = self.accuracy(result, labels)
                # print('the predict is:\n {} \nand the label is:\n {}'.format(self.get_prediction(result[1:6]), self.get_label(labels[:, 1:6])))
                accuracies.append(accuracy)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            # self.save_graph(session)
            # self.read_graph(test_samples[1])

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])

    def get_prediction(self, predictions):
        label_array = np.argmax(predictions, 2).T
        return self.get_label(label_array)

    def label(self, label):
        label = label[label < 10]
        return int(''.join(str(x) for x in label))

    def get_label(self, labels):
        output = []
        for label in labels:
            label = label[label < 10]
            label = int(''.join(str(x) for x in label))
            output.append(label)
        return output

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        filter_map = tensor[-1]
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        self.test_summaries.append(tf.image_summary(name, tensor=filter_map, max_images=how_many))

    def read_graph(self, image, graph_path='./model/svhn-graph.pb'):

        with gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            result = tf.import_graph_def(graph_def,
                                         input_map={'input': image},
                                         return_elements=['output:0'],
                                         name='input')
            sess = tf.Session()
            output = sess.run(result)
            print(output[0])

    def save_graph(self) :
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)

            conv1_weight = self.conv_weights[0].eval(session)
            conv2_weight = self.conv_weights[1].eval(session)
            conv3_weight = self.conv_weights[2].eval(session)
            conv1_biases = self.conv_biases[0].eval(session)
            conv2_biases = self.conv_biases[1].eval(session)
            conv3_biases = self.conv_biases[2].eval(session)

            fc1_weight = self.fc_weights[0].eval(session)
            fc1_biases = self.fc_biases[0].eval(session)

            softmax1_weight = self.softmax_weights[0].eval(session)
            softmax2_weight = self.softmax_weights[1].eval(session)
            softmax3_weight = self.softmax_weights[2].eval(session)
            softmax4_weight = self.softmax_weights[3].eval(session)
            softmax5_weight = self.softmax_weights[4].eval(session)
            softmax6_weight = self.softmax_weights[5].eval(session)
            softmax1_biases = self.softmax_biases[0].eval(session)
            softmax2_biases = self.softmax_biases[1].eval(session)
            softmax3_biases = self.softmax_biases[2].eval(session)
            softmax4_biases = self.softmax_biases[3].eval(session)
            softmax5_biases = self.softmax_biases[4].eval(session)
            softmax6_biases = self.softmax_biases[5].eval(session)
            session.close()

            graph = tf.Graph()
            with graph.as_default():
                input = tf.placeholder("float", shape=[1, 1024], name="input")
                input = tf.reshape(input, [-1, 32, 32, 1])
                conv1_weight_new = tf.constant(conv1_weight, name="constant_conv1_weight")
                conv1_biases_new = tf.constant(conv1_biases, name="constant_conv1_biases")
                output = tf.nn.conv2d(input, filter=conv1_weight_new, strides=[1, 1, 1, 1], padding='VALID') + conv1_biases_new
                output = tf.nn.relu(output)
                output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                conv2_weight_new = tf.constant(conv2_weight, name="constant_conv2_weight")
                conv2_biases_new = tf.constant(conv2_biases, name="constant_conv2_biases")
                output = tf.nn.conv2d(output, filter=conv2_weight_new, strides=[1, 1, 1, 1], padding='VALID') + conv2_biases_new
                output = tf.nn.relu(output)
                output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                conv3_weight_new = tf.constant(conv3_weight, name="constant_conv3_weight")
                conv3_biases_new = tf.constant(conv3_biases, name="constant_conv3_biases")
                output = tf.nn.conv2d(output, filter=conv3_weight_new, strides=[1, 1, 1, 1], padding='VALID') + conv3_biases_new
                output = tf.nn.relu(output)

                fc1_weight_new = tf.constant(fc1_weight, name="constant_fc1_weight")
                fc1_biases_new = tf.constant(fc1_biases, name="constant_fc1_biases")
                output = tf.reshape(output, [-1, 96])
                output = tf.matmul(output, fc1_weight_new) + fc1_biases_new
                output = tf.nn.relu(output)

                softmax1_weight_new = tf.constant(softmax1_weight, name="constant_softmax1_weight")
                softmax1_biases_new = tf.constant(softmax1_biases, name="constant_softmax1_biases")
                softmax2_weight_new = tf.constant(softmax2_weight, name="constant_softmax2_weight")
                softmax2_biases_new = tf.constant(softmax2_biases, name="constant_softmax2_biases")
                softmax3_weight_new = tf.constant(softmax3_weight, name="constant_softmax3_weight")
                softmax3_biases_new = tf.constant(softmax3_biases, name="constant_softmax3_biases")
                softmax4_weight_new = tf.constant(softmax4_weight, name="constant_softmax4_weight")
                softmax4_biases_new = tf.constant(softmax4_biases, name="constant_softmax4_biases")
                softmax5_weight_new = tf.constant(softmax5_weight, name="constant_softmax5_weight")
                softmax5_biases_new = tf.constant(softmax5_biases, name="constant_softmax5_biases")
                softmax6_weight_new = tf.constant(softmax6_weight, name="constant_softmax6_weight")
                softmax6_biases_new = tf.constant(softmax6_biases, name="constant_softmax6_biases")

                length = tf.matmul(output, softmax1_weight_new) + softmax1_biases_new
                number1 = tf.matmul(output, softmax2_weight_new) + softmax2_biases_new
                number2 = tf.matmul(output, softmax3_weight_new) + softmax3_biases_new
                number3 = tf.matmul(output, softmax4_weight_new) + softmax4_biases_new
                number4 = tf.matmul(output, softmax5_weight_new) + softmax5_biases_new
                number5 = tf.matmul(output, softmax6_weight_new) + softmax6_biases_new
                logits = tf.pack([
                    tf.nn.softmax(length),
                    tf.nn.softmax(number1),
                    tf.nn.softmax(number2),
                    tf.nn.softmax(number3),
                    tf.nn.softmax(number4),
                    tf.nn.softmax(number5)])

                result = tf.argmax(logits[1:6], 2)
                result = tf.cast(result, tf.int32)
                output = tf.transpose(result)
                output = tf.reshape(output, [-1])
                tf.cast(output, tf.float32, name="output")

                graph_default = graph.as_graph_def()
                tf.train.write_graph(graph_default, './model', 'svhn-graph.pb', as_text=False)
