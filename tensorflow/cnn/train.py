if __name__ == '__main__':
    from network import Network

    def read_image(image_file):
        import pickle
        try:
            with open(image_file, "rb") as f:
                unpickler = pickle.Unpickler(f)
                images = unpickler.load()
                # print(len(images['dataset']), len(images['labels']))
                return images['dataset'], images['labels']
        except Exception as e:
            print('Unable to read data from', image_file, ':', e)
            raise

    train_samples, train_labels = read_image('../svhn/pickle/train-n.pickle')
    test_samples, test_labels = read_image('../svhn/pickle/test-n.pickle')

    train_labels = train_labels[:, 0]
    test_labels = test_labels[:, 0]

    print('Train Set', train_samples.shape, train_labels.shape)
    print('Test Set', test_samples.shape, test_labels.shape)

    # image_size = load.image_size
    # num_labels = load.num_labels
    # num_channels = load.num_channels

    image_size = 32
    num_labels = 10
    num_channels = 1

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
        stepStart = 0
        i = 0
        while i < iteration_steps:
            stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
            yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
            i += 1


    net = Network(train_batch_size=64, test_batch_size=500, pooling_scale=2, dropout_rate=0.9, base_learning_rate=0.001, decay_rate=0.99)
    net.define_inputs(
        train_samples_shape=(64, image_size, image_size, num_channels),
        train_labels_shape=(64, num_labels),
        test_sample_shape=(500, image_size, image_size, num_channels)
    )
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

    net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=32, activation='relu', name='fc1')
    net.add_fc(in_num_nodes=32, out_num_nodes=10, activation=None, name='fc2')

    # Average Accuracy: 90.7769230769
    # Standard Deviation: 1.56777836255

    net.define_model()
    # net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator, iteration_steps=3000, test_data_iterator=test_data_iterator)
    # net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=500)
    net.test(test_samples, test_labels, data_iterator=test_data_iterator)

else:
    raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')
