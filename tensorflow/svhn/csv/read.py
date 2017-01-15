import numpy as np
import sys


def read_image(image_file):
    import pickle
    try:
        with open(image_file, "rb") as f:
            unpickler = pickle.Unpickler(f)
            images = unpickler.load()
            print(len(images['dataset']), len(images['labels']))
            return images['dataset'], images['labels']
    except Exception as e:
        print('Unable to read data from', image_file, ':', e)
        raise


def draw(image):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)
    plt.show()


def save_image(csv_file, image_dir='/Users/zpwu/Downloads/train'):
    import pandas as pd
    from scipy.misc import imread, imresize
    import os
    print(image_dir)
    df = pd.read_csv(csv_file)
    dataset = np.ndarray(shape=(df.shape[0], 224, 224, 3), dtype=np.float32)
    labels = np.ndarray(shape=(df.shape[0], 5, 10), dtype=np.float32)
    # length = np.ndarray(shape=(df.shape[0], 10), dtype=np.float32)
    for index, row in df.iterrows():
        if os.path.isfile(os.path.join(image_dir, row['FileName'])):
            img = imread(os.path.join(image_dir, row['FileName']))
            img_tinted = img * [1, 0.95, 0.9]
            img_tinted = imresize(img_tinted, (224, 224))
            dataset[index, :, :, :] = img_tinted


            image_label = row['DigitLabel']
            # length[index] =
            label = [len(str(image_label))]
            for i in str(image_label):
                label.append(int(i))

            label = np.array(label)
            label.shape = label.shape[0]
            label = (np.arange(10) == label[:, None]).astype(np.float)

            other = np.full((6 - label.shape[0], 10), 0.1)
            label=np.concatenate((label, other), axis=0)


            print('transfer {0} to vector {1}'.format(image_label, label))
            labels[index] = label
    return dataset, labels

def normalize(samples):
    pixel_sum = np.add.reduce(samples, keepdims=True, axis=3)
    pixel_average = pixel_sum / 3.0  # save memory and speed up training
    return pixel_average / 128.0 - 1.0

def save(dataset, labels, pickleFile):
    import pickle

    dataset = normalize(dataset)

    try:
        with open(pickleFile, 'wb') as f:
            save = {
                'dataset': dataset,
                'labels': labels
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to data.pickle :', e)
        raise


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("    {0} <inputCsvFilePath> <outputPickleFilePath> <imagesFilePath>".format(sys.argv[0]))
        return

    inputCsvFilePath = sys.argv[1]
    outputPickleFilePath = sys.argv[2]
    imagesFilePath = sys.argv[3]
    print("Converting {0} to {1} from {2}".format(inputCsvFilePath, outputPickleFilePath, imagesFilePath))

    dataset, labels = save_image(inputCsvFilePath, imagesFilePath)
    print(dataset.shape, labels.shape)
    save(dataset, labels, outputPickleFilePath)


if __name__ == '__main__':
    main()
