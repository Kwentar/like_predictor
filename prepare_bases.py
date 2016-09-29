import cv2
import os
from shutil import rmtree


def write_weights_in_file(weights: dict, file_name: str):
    with open(file_name, 'a+') as f:
        for key, value in weights.items():
            f.write(key.__str__() + ";" + value.__str__() + "\n")


def read_weights_from_file(file_name: str):
    weights = {}
    with open(file_name) as f:
        lines = list(map(str.strip, f.readlines()))
    if lines:
        split_values = list(map(lambda x: x.split(';'), lines))
        weights = dict((x[0], float(x[1])) for x in split_values)
    return weights


def resize_and_copy_images(dir_, dst_dir_, current_index, weight_file_name='all_weights.txt', dir_for_images='images'):
    weights = read_weights_from_file(os.path.join(dir_, 'weights.txt'))
    image_names = os.listdir(dir_)
    image_exts = ['.jpg', '.png']
    image_names = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_exts, image_names))
    image_weights = dict()
    for image_name in image_names:
        image_weight = weights[image_name]
        image = cv2.imread(os.path.join(dir_, image_name))
        new_image = cv2.resize(image, (256, 256))
        if new_image.shape:
            image_file_name = str(current_index) + '.png'
            cv2.imwrite(os.path.join(os.path.join(dst_dir_, dir_for_images), image_file_name), new_image)
            image_weights[image_file_name] = image_weight
            current_index += 1
    write_weights_in_file(image_weights, os.path.join(dst_dir_, weight_file_name))
    return current_index


def calc_weights(dir_):
    if not dir_: 
        return
    uid = dir_.split('/')[-1]
    with open(os.path.join(dir_, uid + '.txt')) as file_with_likes:
        lines = map(str.strip, file_with_likes.readlines())
    if lines is not None:
        split_values = list(map(lambda x: x.split(','), lines))
        likes = dict((x[0], int(x[1])) for x in split_values)
        max_value = max(likes.values())
        if max_value:
            for key, _ in likes.items():
                likes[key] /= max_value
            write_weights_in_file(likes, os.path.join(dir_, "weights.txt"))


def test(dst_dir_, weight_file_name='all_weights.txt', dir_for_images='images'):
    weights = read_weights_from_file(os.path.join(dst_dir_, weight_file_name))
    image_names = os.listdir(os.path.join(dst_dir_, dir_for_images))
    image_exts = ['.jpg', '.png']
    image_names = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_exts, image_names))
    if len(image_names) != len(weights):
        print('Alert: images count is {}, weights count is: {}'.format(len(image_names), len(weights)))
        for image_name in image_names:
            if image_name in weights.keys():
                del weights[image_name]
            else:
                print('Alert: {} not in weights'.format(image_name))
        for key in weights.keys():
            print('Alert: {} weight without image'.format(key))
        return
    print('Test has been finished, all is ok')


def clear_previous_dir(dir_):
    for item in map(lambda x: os.path.join(dir_, x), os.listdir(dir_)):
        if os.path.isdir(item):
            rmtree(item)
        else:
            os.remove(item)


def run():
    src_dir = '/home/kwent/Bases/vk'
    dst_dir = '/home/kwent/Bases/vk_likes'
    dst_dir_for_images = 'images'
    dst_weight_file_name = 'all_weights.txt'
    dirs = list(map(lambda x: os.path.join(src_dir, x), os.listdir(src_dir)))
    dirs = list(filter(os.path.isdir, dirs))
    count_person = len(dirs)
    index = 1
    image_index = 1
    limit = 0
    print('Removing previous version...')
    clear_previous_dir(dst_dir)
    print('Starting conversion...')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(os.path.join(dst_dir, dst_dir_for_images)):
        os.makedirs(os.path.join(dst_dir, dst_dir_for_images))
    for current_dir in dirs:
        image_index = resize_and_copy_images(current_dir,
                                             dst_dir,
                                             image_index,
                                             weight_file_name=dst_weight_file_name,
                                             dir_for_images=dst_dir_for_images)
        if index % 10 == 0:
            print(index, '/', count_person, '|', round(index/count_person*100, 5), '% images: ', image_index)
        index += 1
        if limit and image_index >= limit:
            break
    print('Conversion has been finished!')
    test(dst_dir_=dst_dir, weight_file_name=dst_weight_file_name, dir_for_images=dst_dir_for_images)

# run()
