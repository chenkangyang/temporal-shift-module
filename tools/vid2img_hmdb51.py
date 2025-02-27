"""
# data_pth/NonViolence/a.mp4 -----> data_pth/img/NonViolence/a/%06d.jpg
# data_pth/Violence/a.mp4 -----> data_pth/img/Violence/a/%06d.jpg

recv 2 args: input_path, output_path
"""
from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import random
import pandas as pd

n_thread = 100

def vid2jpg(file_name, class_path, dst_class_path):
    # print(file_name)
    
    if '.mp4' not in file_name:
        if '.avi' not in file_name:
            print("skip", file_name)
            return

    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)

    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_000001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
        return
    cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%06d.jpg\"'.format(video_file_path, dst_directory_path)
    # print(cmd)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def class_process(dir_path, dst_dir_path, class_name):
    print('*' * 20, class_name, '*'*20)
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        print('*** is not a dir {}'.format(class_path))
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)

    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    vid_list = os.listdir(class_path)
    vid_list.sort()
    p = Pool(n_thread)
    from functools import partial
    worker = partial(vid2jpg, class_path=class_path, dst_class_path=dst_class_path)

    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    
    class_list = os.listdir(dir_path)
    class_list.sort()
    for class_name in class_list:
        class_process(dir_path, dst_dir_path, class_name)