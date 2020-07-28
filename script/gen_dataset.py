import os
import pandas as pd


data_folder = '/mnt/projects/CT_Dental/data_v1'
image_folder = []
for file in os.listdir(data_folder):
    if file.startswith('case'):
        dropped_cases_idx = [33, 45, 53, 55, 84, 88, 89, 104, 106, 108, 109]
        dropped_cases_name = ['case_{}'.format(idx) for idx in dropped_cases_idx]
        is_dropped = False
        for case_name in dropped_cases_name:
            if file.startswith(case_name):
                is_dropped = True

        if not is_dropped:
            image_folder.append(file)

image_folder.sort()
num_training = int(len(image_folder)*0.8)

training_set = image_folder[:num_training]
testing_set = image_folder[num_training:]

server_data_folder = '/shenlab/lab_stor6/projects/CT_Dental/data'
train_content = []
for case in training_set:
    train_content.append([case, os.path.join(server_data_folder, case, 'org.mha'), os.path.join(server_data_folder, case, 'seg.mha')])

train_df = pd.DataFrame(train_content, columns=['image_name', 'image_path', 'mask_path'])
train_df.to_csv('/home/ql/debug/train.csv')

test_content = []
for case in testing_set:
    test_content.append([case, os.path.join(server_data_folder, case, 'org.mha'), os.path.join(server_data_folder, case, 'seg.mha')])

test_df = pd.DataFrame(test_content, columns=['image_name', 'image_path', 'mask_path'])
test_df.to_csv('/home/ql/debug/test.csv')
