import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", type=str)
    parser.add_argument("dst_file", type=str)
    args = parser.parse_args()

    f = open(args.dst_file, 'wt')
    for _, _, files in os.walk(args.img_dir):
        for img_file in files:
            file_name = img_file.split('.')[0]
            f.write(file_name + '\n')
    f.close()