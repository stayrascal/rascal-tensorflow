from cnn import Network


def draw(image):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image.reshape(32, 32))
    plt.show()


def read_image(image_file):
    import pickle
    try:
        with open(image_file, 'rb') as f:
            save = pickle.load(f)
            X_train = save['train_dataset']
            y_train = save['train_labels']
            X_val = save['valid_dataset']
            y_val = save['valid_labels']
            X_test = save['test_dataset']
            y_test = save['test_labels']
            del save
            print('Training data shape:', X_train.shape)
            print('Training label shape:', y_train.shape)
            print('Validation data shape:', X_val.shape)
            print('Validation label shape:', y_val.shape)
            print('Test data shape:', X_test.shape)
            print('Test label shape:', y_test.shape)
            return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        print('Unable to read data from', image_file, ':', e)
        raise


def test_data_iterator(samples, labels, chunkSize):
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd


def train_data_iterator(samples, labels, iteration_steps, chunkSize):
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    i = 0
    while i < iteration_steps:
        stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
        yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
        i += 1


train_samples, train_labels, validation_samples, validation_labels, test_samples, test_labels = read_image('SVHN.pickle')

print('Train Set', train_samples.shape, train_labels.shape)
print('Validation Set', validation_samples.shape, validation_labels.shape)
print('Test Set', test_samples.shape, test_labels.shape)

# We processed image size to be 32
image_size = 32
# Number of channels: 1 because greyscale
num_channels = 1
# Mini-batch size
batch_size = 16
# Number of output labels
num_labels = 11

# depth: number of filters (output channels) - should be increasing
# num_channels: number of input channels set at 1 previously
patch_size = 5
depth_1 = 16
depth_2 = depth_1 * 2
depth_3 = depth_2 * 3

# Number of hidden nodes in fully connected layer 1
num_hidden = 64
shape = [batch_size, image_size, image_size, num_channels]

net = Network(train_batch_size=batch_size, test_batch_size=500, pooling_scale=2, dropout_rate=0.9, base_learning_rate=0.001, decay_rate=0.99)
net.define_inputs(
    train_samples_shape=(batch_size, image_size, image_size, num_channels),
    test_samples_shape=(500, image_size, image_size, num_channels),
    train_labels_shape=(batch_size, 6)
)
net.add_conv(patch_size=5, in_depth=num_channels, out_depth=16, activation='relu', pooling=True, name='conv1')
net.add_conv(patch_size=5, in_depth=16, out_depth=32, activation='relu', pooling=True, name='conv2')
net.add_conv(patch_size=5, in_depth=32, out_depth=96, activation='relu', pooling=False, name='conv3')
# net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')
#
net.add_fc(in_num_nodes=96, out_num_nodes=64, activation='relu', name='fc1')
# # net.add_fc(in_num_nodes=32, out_num_nodes=10, activation=None, name='fc2')
#
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='length')
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='softmax1')
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='softmax2')
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='softmax3')
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='softmax4')
net.add_softmax_fc(in_num_nodes=64, out_num_nodes=11, activation='relu', name='softmax5')
# net.add_softmax_fc(in_num_nodes=32, out_num_nodes=10, activation='softmax', name='softmax6')
#
# # Average Accuracy: 90.7769230769
# # Standard Deviation: 1.56777836255
#
net.define_model()

# Without regularization                     With regularization
# Minibatch accuracy: 86.2%
# Average Accuracy: 89.6461397059            Average Accuracy: 88.7913602941
# Standard Deviation: 3.71656010905          Standard Deviation: 3.89332582983

# net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator, iteration_steps=15001, test_data_iterator=test_data_iterator)
# net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=14001)
net.test(test_samples, test_labels, data_iterator=test_data_iterator)
# net.test(validation_samples, validation_labels, data_iterator=test_data_iterator)
# net.save_graph()
# net.read_graph(test_samples[0])