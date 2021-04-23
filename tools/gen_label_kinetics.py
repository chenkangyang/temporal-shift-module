# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

import os
import json
import pandas as pd

dataset_path = '../data/kinetics/images240/'
label_path = '../data/kinetics/'

if __name__ == '__main__':
    with open('kinetics_label_map.txt') as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
    assert len(set(categories)) == 400
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    # print(dict_categories)

    files_input = '../data/kinetics/kinetics.json'
    # files_output = ['val_videofolder.txt', 'train_videofolder.txt']
    files_output = ['train_videofolder.txt', 'val_videofolder.txt', 'test_videofolder.txt', 'missing_train_videofolder.txt', 'missing_val_videofolder.txt', 'missing_test_videofolder.txt']

    with open(files_input) as f:
        data = json.load(f)

    labels = data['labels']
    database = data['database']

    dict_categories = {}
    for i, category in enumerate(labels):
        dict_categories[category] = i

    assert len(set(dict_categories)) == 400

    train_list = []
    missing_train_list = []
    val_list = []
    missing_val_list = []
    test_list = []
    missing_test_list = []


    train_df = pd.DataFrame(columns=['path', 'num', 'label'])
    missing_train_df = pd.DataFrame(columns=['path', 'label'])
    val_df = pd.DataFrame(columns=['path', 'num', 'label'])
    missing_val_df = pd.DataFrame(columns=['path', 'label'])
    test_df = pd.DataFrame(columns=['path'])
    missing_test_df = pd.DataFrame(columns=['path'])



    processed_num = 0
    for key, value in database.items():
        video_name = key
        video_annotations = value['annotations']
        cur_subset = value['subset']
        if video_annotations:
            cur_label = video_annotations['label']
            cur_path = os.path.join(cur_label, video_name)
            cur_label_idx = dict_categories[cur_label]
            line = {}
            if not 'segment' in video_annotations.keys():
                if not os.path.exists(os.path.join(dataset_path, cur_path)): # 无segment且视频不存在, 等待后续被写入missing文件
                    pass
                else: # 视频存在但是，没有segment, 计算 segment, 更新json; validation都没有segment
                    dir_files = os.listdir(os.path.join(dataset_path, cur_path))
                    cur_num = len(dir_files)
                    line = {'path': cur_path, 'num': cur_num, 'label': cur_label_idx}
                    data['database'][video_name][video_annotations]['segment'] = [1, cur_num+1]
                    import pdb; pdb.set_trace() # ! 没有执行到这里，说明无segment的视频（validation）都不存在
            else:
                cur_segment = video_annotations['segment']
                cur_num = cur_segment[1] - cur_segment[0]
                line = {'path': cur_path, 'num': cur_num, 'label': cur_label_idx}

            if cur_subset == "training":
                if not os.path.exists(os.path.join(dataset_path, cur_path)):
                    missing_train_list.append({'path': cur_path, 'label': cur_label_idx})
                else:
                    train_list.append(line)
            elif cur_subset == "validation":
                
                if not os.path.exists(os.path.join(dataset_path, cur_path)):
                    missing_val_list.append({'path': cur_path, 'label': cur_label_idx})
                else:
                    print("val", line)
                    val_list.append(line)
        else: # 无标注, 都是testing测试视频
            if cur_subset == "testing":
                test_vid_path = os.path.join("test", video_name)
                if not os.path.exists(os.path.join(dataset_path, test_vid_path)):
                    missing_test_list.append({'path': test_vid_path})
                else:
                    test_list.append({'path': test_vid_path})
            else: 
                print("奇怪的事情发生了：无标注的视频", video_name)
                raise FileNotFoundError
        
        processed_num += 1
    
        if processed_num % 1000 == 0:
            print("{} videos processed".format(processed_num))
    
    label_train_path = os.path.join(label_path, files_output[0])
    label_val_path = os.path.join(label_path, files_output[1])
    label_test_path = os.path.join(label_path, files_output[2])
    missing_train_path = os.path.join(label_path, files_output[3])
    missing_val_path = os.path.join(label_path, files_output[4])
    missing_test_path = os.path.join(label_path, files_output[5])
    
    train_df = pd.DataFrame(train_list)
    missing_train_df = pd.DataFrame(missing_train_list)
    val_df = pd.DataFrame(val_list)
    missing_val_df = pd.DataFrame(missing_val_list)
    test_df = pd.DataFrame(test_list)
    missing_test_df = pd.DataFrame(missing_test_list)
    
    train_df.to_csv(label_train_path, encoding='utf-8', index=False, header=False)
    print("{} train label saved in {}".format(train_df.shape[0], label_train_path))
    val_df.to_csv(label_val_path, encoding='utf-8', index=False, header=False)
    print("{} val label saved in {}".format(val_df.shape[0], label_val_path))
    test_df.to_csv(label_test_path, encoding='utf-8', index=False, header=False)
    print("{} test label saved in {}".format(test_df.shape[0], label_test_path))
    missing_train_df.to_csv(missing_train_path, encoding='utf-8', index=False, header=False)
    print("{} missing train videos saved in {}".format(missing_train_df.shape[0], missing_train_path))
    missing_val_df.to_csv(missing_val_path, encoding='utf-8', index=False, header=False)
    print("{} missing val videos saved in {}".format(missing_val_df.shape[0], missing_val_path))
    missing_test_df.to_csv(missing_test_path, encoding='utf-8', index=False, header=False)
    print("{} missing val videos saved in {}".format(missing_test_df.shape[0], missing_test_path))