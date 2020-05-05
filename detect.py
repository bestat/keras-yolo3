import sys
import os
import csv
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

def detect_img(yolo, image_path):
    # while True:
    # try:
    image = Image.open(image_path)
    r_image = yolo.detect_image(image)
    r_image.show()
    yolo.close_session()


def detect_img_on_csv(yolo, csv_path, save_root='./output'):
    """
    test.txtの情報を用いて、テスト画像すべてについて推論する
    """
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            image = Image.open(row[0])
            r_image = yolo.detect_image(image)
            name, ext = os.path.splitext(os.path.basename(row[0]))
            savename = os.path.join(save_root, name + '_detect' + ext)
            r_image.save(savename)
        yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='webcam',
        help = "csv path"
    )
    parser.add_argument(
        "--saveroot", nargs='?', type=str,required=False,default='./output',
        help = "save path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        if 0:
            detect_img(YOLO(**vars(FLAGS)), FLAGS.input)
        else:
            detect_img_on_csv(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.saveroot)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
