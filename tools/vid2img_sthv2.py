# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading
import time

NUM_THREADS = 100

# ln -s /mnt/data/kychen/datasets/tsm_data data
PROJECT_ROOT = '/mnt/data/kychen/workspace/temporal-shift-module'
VIDEO_ROOT = os.path.join(PROJECT_ROOT, 'data/something/v2/20bn-something-something-v2')         # Downloaded webm videos
FRAME_ROOT = os.path.join(PROJECT_ROOT, 'data/something/v2/20bn-something-something-v2-frames')  # Directory for extracted frames



def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    date = time.strftime("%Y-%m-%d", time.localtime())
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg >2 {}_ff.log\"'.format(VIDEO_ROOT, video,
                                                                                                FRAME_ROOT, video[:-5], date)
    os.system(cmd)


def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    splits = list(split(video_list, NUM_THREADS))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()