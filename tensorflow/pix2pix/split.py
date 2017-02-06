from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True,
                    help="path to folder containing images")
parser.add_argument("--train_frac", type=float, default=0.8,
                    help="percentage of images to use for trainning set")
parser.add_argument("--test_frac", type=float,
                    help="percentage of images to use for test set")
a = parser.parse_args()


def main():
    random.seed(0)

    files = glob.glob(os.path.join(a.dir, "*.png"))
    assigments = []
    assigments.extend(["train"] * int(a.train_frac * len(files)))
    assigments.extend(["test"] * int(a.test_frac * len(files)))
    assigments.extend(["val"] * int(len(files) - len(assigments)))
    random.shuffle(assigments)

    for name in ["train", "val", "test"]:
        if name in assigments:
            d = os.path.join(a.dir, name)
            if not os.path.exists(d):
                os.makedirs(d)

    print(len(files), len(assigments))
    for inpath, assigment in zip(files, assigments):
        outpath = os.path.join(a.dir, assigment, os.path.basename(inpath))
        print(inpath, "->", outpath)
        os.rename(inpath, outpath)

main()
