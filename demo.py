# @Time    : 2018/2/7
# @Author  : fh
# @File    : demo.py
# @Desc    :
"""
    Give a demo to recognize plate from a raw image
"""
from lib.config import cfg_from_file, cfg
from lib.detector import detect as plate_detect
from lib.recognizer import recognize as chars_recognize
from lib.utils.align import align
from models.svm.process_images import process_image
from models.svm.char_recognition import prediction
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from lib.mrcnn.plate_detect import PlateDetect

plt.rcParams['font.sans-serif'] = ['SimHei']  # display chinese title in plt
plt.rcParams['axes.unicode_minus'] = False  # display minus normally


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='ULPR demo')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file', default=None, type=str)
    # parser.add_argument('--path', dest='image_path',
    #                     help='image path', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    start = time.clock()
    src = cv2.imread("data/demo/0.jpg")
    if cfg.VIS:
        plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        plt.show()
    # plate = PlateDetect()
    # results = plate.detect(str(cfg.OUTPUT_DIR))
    results = plate_detect(src)

    if results is not None:
        for res in results:
            print("Plate position: \n", res)
            vis_image = align(src, res)
            # img = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            # if img is not None:
            #     plt.title("card")
            #     plt.imshow(img)
            #     plt.show()
            #     images = process_image(img)
            #     prediction(images)
            rec_res = chars_recognize(vis_image)
            print("Chars Recognize: ", rec_res)
            if cfg.VIS:
                plt.title(rec_res)
                plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                plt.show()
        end = time.clock()
        print("Predtion spend {:.3f}".format((end-start)))

