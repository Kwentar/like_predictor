import tensorflow as tf
import os
import numpy as np
import time
import cv2
from prepare_bases import read_weights_from_file, clear_previous_dir


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))


tf.app.flags.DEFINE_string('directory', 'converted_data',
                           'Directory to write the '
                           'converted result')
# tf.app.flags.DEFINE_integer('validation_size', 10000,
#                             'Number of examples to separate from the training '
#                             'ckpt for the validation set.')
FLAGS = tf.app.flags.FLAGS


def convert_to(images, labels, name, part_index):
    """
    Conver images into tfrecord file
    :param images: numpy array with shape [num_examples, h, w, c]
    :param labels: numpy array with shape [num_examples], float
    :param name: name of file
    :param part_index: index of part
    :return: write file with name "name + index + .tfrecords" in FLAGS.dircetory
    """
    num_examples = labels.shape[0]
    print('labels shape is ', labels.shape[0])
    if images.shape[0] != num_examples:
        raise ValueError("Images size {} does not match label size {}.".format(images.shape[0], num_examples))

    filename = os.path.join(FLAGS.directory, name + str(part_index) + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(
                float_list=tf.train.FloatList(value=[labels[index]])),
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_raw]))}))
        writer.write(example.SerializeToString())


def read_and_decode(filename_queue, image_shape):
    """
    read files from tf.train.string_input_producer
    :param filename_queue: queue of files
    :param image_shape: image shape, i.e. (256, 256, 3)
    :return: image as tf.float32 array in [-0.5 0.5] and label as tf.float32
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'label': tf.FixedLenFeature([], tf.float32),
                                                 'image': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.reshape(image, image_shape)
    image.set_shape(image_shape)

    image = tf.cast(image, tf.float32)/255 - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.float32)

    return image, label


def write_data_to_tf_record(images_dir, labels_file):
    """
    Create folder FLAGS.directory with tf.record files
    :param images_dir: dir which contains images, i.e.: 1.png, 2.png etc.
    :param labels_file: txt file, each line is image_name;label, i.e.; 1.png:0.151132
    :return: None
    """
    if not os.path.exists(FLAGS.directory):
        os.makedirs(FLAGS.directory)
    clear_previous_dir(FLAGS.directory)
    images_per_file = 16384
    index = 0
    image_names = os.listdir(images_dir)
    count_images = len(image_names)
    weights = read_weights_from_file(labels_file)

    while index * images_per_file < count_images:
        curr_images = list()
        tmp_weights = list()
        with Profiler() as p:
            print('Staring {} file...'.format(index))
            for i in range(index * images_per_file, (index + 1) * images_per_file):
                if i % 1000 == 0:
                    print('Prepare {} %'.format(i / count_images * 100))
                image = cv2.imread(os.path.join(images_dir, os.path.join(images_dir, image_names[i])))
                if image.shape:
                    curr_images.append(image)
                    tmp_weights.append(weights[image_names[i]])
            convert_to(np.array(curr_images), np.array(tmp_weights), 'test', index)
        index += 1


def show_samples():
    """
    Show one image from each tfrecords file
    :return: None
    """
    files = os.listdir(FLAGS.directory)
    for file in files:
        image, label = read_and_decode(tf.train.string_input_producer([os.path.join(FLAGS.directory, file)]),
                                       (256, 256, 3))
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        label_val_1, image_val_1 = sess.run([label, image])

        cv2.imshow('s', (image_val_1 + 0.5))
        print(label_val_1)
        cv2.waitKey(1000)

# write_data_to_tf_record('/home/kwent/Bases/vk_likes/images/', '/home/kwent/Bases/vk_likes/all_weights.txt')

