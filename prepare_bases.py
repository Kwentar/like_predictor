import cv2
import os


def write_weights_in_file(weights: dict, file_name: str):
    with open(file_name, 'w+') as f:
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


def resize_and_copy_images(dir_, dst_dir, weight_file_name='all_weights.txt', dir_for_images='images'):
    current_index = 1
    # Todo: get index
    weights = read_weights_from_file(os.path.join(dir_, 'weights.txt'))
    image_names = os.listdir(dir_)
    image_exts = ['.jpg', '.png']
    image_names = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_exts, image_names))
    for image_name in image_names:
        image_weight = weights[image_name]
        image = cv2.imread(image_name)
        new_image = cv2.resize(image, (256, 256))
        if new_image.shape:
            # Todo save image in dir_for_images with name current_index.png and save weight of image in weight_file_name
            pass


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


src_dir = '/media/kwent/A278648A78645F53/vk'
dst_dir = '/home/kwent/bases/vk_likes'
dirs = list(map(lambda x: os.path.join(src_dir, x), os.listdir(src_dir)))
dirs = list(filter(os.path.isdir, dirs))
count_person = len(dirs)
index = 1
for current_dir in dirs:
    resize_and_copy_images(current_dir, dst_dir)
    if index % 100 == 0:
        print(index, '/', count_person, '|', round(index/count_person*100, 5), '%')
    index += 1
