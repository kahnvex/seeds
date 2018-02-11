import time
import os
import argparse

from sklearn.model_selection import train_test_split
from shutil import copyfile


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--validation-split', '-s', type=float, default=0.15)

    return parser.parse_args()


def main(args):
    otrain_dir = 'data/original/train'
    train_dir = 'data/train'
    val_dir = 'data/validation'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    val_split = args.validation_split

    for dirty in os.listdir(otrain_dir):
        imgs = os.listdir('%s/%s' % (otrain_dir, dirty))
        train_imgs, val_imgs = train_test_split(imgs, test_size=val_split,
                                                random_state=int(time.time()))
        os.makedirs('%s/%s' % (train_dir, dirty), exist_ok=True)
        os.makedirs('%s/%s' % (val_dir, dirty), exist_ok=True)

        for img in train_imgs:
            copyfile('%s/%s/%s' % (otrain_dir, dirty, img),
                     '%s/%s/%s' % (train_dir, dirty, img))

        for img in val_imgs:
            copyfile('%s/%s/%s' % (otrain_dir, dirty, img),
                     '%s/%s/%s' % (val_dir, dirty, img))


if __name__ == '__main__':
    main(get_args())
