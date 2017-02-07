from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True,
                    help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True,
                    help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int,
                    help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=10,
                    help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50,
                    help="display progress every progress_freq steps")
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
parser.add_argument("--trace_freq", type=int, default=0,
                    help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000,
                    help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0,
                    help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split A image into brightness (A) and color (B), ignore B image")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images in batch")
parser.add_argument("--which_direction", type=str,
                    default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64,
                    help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64,
                    help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286,
                    help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true",
                    help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip",
                    action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002,
                    help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5,
                    help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0,
                    help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0,
                    help="weight on GAN term for generator gradient")
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple(
    "Model", "outputs, predict_real, predict_real, predict_fake, discrim_loss, gen_loss_GAN, gen_loss_L1, train")


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.getshape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels],
                                 dtype=tf.float32, initialize=tf.random_normal_initializer(0, 0.02))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels] => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                              1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [
                            1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x) / 2
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable(
            "offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer)
        scale = tf.get_variable("scale", [
                                channels], dtype=tf.float32, initializer=tf.random_normal_initializer(10, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.monments(
            input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [
            int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filter, [
                                      batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 2], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("Image must be either 3 or 4 dimensions")

    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + \
                (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X         Y         Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = F(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(
                xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <=
                                  (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(
                xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4 / 29) * \
                linear_mask + (xyz_normalized_pixels **
                               (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0,  500.0,    0.0],  # fx
                [116.0, -500.0,  200.0],  # fy
                [0.0,    0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(
                fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0,  1 / 116.0],  # l
                [1 / 500.0,     0.0,      0.0],  # a
                [0.0,     0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(
                lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(
                fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4 / 29)) * \
                linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

            with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660,  0.0556434],  # x
                [-1.5371385,  1.8760108, -0.2040259],  # y
                [-0.4985314,  0.0415560,  1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(
                rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + \
                ((rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_example():
    if not os.path.exists(a.input_dir):
        raise Exception("Input dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.encode_jpeg

    if len(input_paths) == 0:
        input_path = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("Input dir doesn't contain image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for in input_paths):
        input_paths = sorted(input_path, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(
            input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(
            tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies(assertion):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
            # black and white with input range [0, 100]
            a_images = tf.expand_dims(L_chan, axis=2) / 50 - 1
            # color channels with input range ~[-110, 110], not exact
            b_images = tf.stack([a_chan, b_chan], axis=2) / 110
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = raw_input[:, :width // 2, :] * 2 - 1
            b_images = raw_input[:, width // 2:, :] * 2 - 1

        if a.which_direction == "AtoB":
            inputs, targets = [a_images, b_images]
        elif a.which_direction == "BtoA":
            inputs, targets = [b_images, a_images]
        else:
            raise Exception("invalid direction")

        # synchronize seed for image operations so that we do the same operations to both
        # input and output images
        seed = random.randint(0, 2**31 - 1)

        def transform(image):
            r = image
            if a.flip:
                r = tf.image.random_flip_left_right(r, seed=seed)

            # area produces a nice downscaling, but does nearest neighbor for upscaling
            # assume we're going to be doing downscaling here
            r = tf.image.resize_images(
                r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

            offset = tf.cast(tf.floor(tf.random_uniform(
                [2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            if a.scale_size > CROP_SIZE:
                r = tf.image.crop_to_bounding_box(
                    r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
            elif a.scale_size < CROP_SIZE:
                raise Exception("scale size cannot be less than crop size")
            return r

        with tf.name_scope("input_images"):
            input_images = transform(inputs)

        with tf.name_scope("target_images"):
            target_images = transform(targets)

        paths, inputs, targets = tf.train.batch(
            [paths, input_images, target_images], batch_size=a.batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

        return Examples(
            paths=paths,
            inputs=inputs,
            targets=targets,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )


