import os
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

def get_boundary_indices(matrix):
    # 找到所有值为1的元素的行号和列号
    rows, cols = np.where(matrix == 1)
    
    # 用于存储边界元素的行号和列号
    boundary_indices = []
    
    # 遍历所有值为1的元素
    for i, j in zip(rows, cols):
        # 检查上下左右四个方向的邻居
        neighbors = [
            (i-1, j),  # 上
            (i+1, j),  # 下
            (i, j-1),  # 左
            (i, j+1),  # 右
        ]
        
        # 如果任何一个邻居是0，则该元素是边界元素
        for ni, nj in neighbors:
            if ni < 0.0 or ni >= matrix.shape[0] or nj < 0.0 or nj >= matrix.shape[1]:
                # 超出边界的邻居视为0
                boundary_indices.append((int(j), int(i)))
                break
            elif matrix[ni, nj] == 0.0:
                boundary_indices.append((int(j), int(i)))
                break
    
    return boundary_indices

startTime = time.time()
count = 0
checkpoints_yolo11x_folder = os.path.join('checkpoints', 'segmentation')
modelType = 'x'
model_path = os.path.join(checkpoints_yolo11x_folder, f"yolo11{modelType}-seg.pt")
model = YOLO(model_path)
forPredict_rootFolder = os.path.join('forPredict')
conferenceTypes = ['person', 'car', 'motorcycle', ] #  'people', 
conferenceType_class = {'motorcycle-electricCar':3,'car':2, 'motorcycle':3,'person':0,} #
confThreshold_folder = {0.3:'0.3',}

for conferenceType in conferenceTypes:
    count = 0
    images_source_folder = os.path.join(forPredict_rootFolder, 'oriImage', conferenceType)
    result_box_folder = os.path.join('forPredict', 'result', 'boxes')
    result_mask_folder = os.path.join('forPredict', 'result', 'masks')
    result_conf_folder = os.path.join('forPredict', 'result', 'confs')

    # noise_txt_path = os.path.join(forPredict_rootFolder, f'{conferenceType}_yolox_0.3_noise.txt')
    # noiseImage_dict = {}
    # with open(noise_txt_path, 'r') as f:
    #     for row in f.readlines():
    #         image_file = row.strip()
    #         noiseImage_dict[image_file] = None

    needImage_dict = {}
    need_images_path = os.path.join('forPredict', 'result', 'images', '0.3', f"{conferenceType}-old")
    for item in os.listdir(need_images_path):
        needImage_dict[item] = None
    complete_dict = {}
    complete_images_path = os.path.join('forPredict', 'result', 'images', '0.3', f"{conferenceType}")
    for item in os.listdir(complete_images_path):
        complete_dict[item] = None

    images_source_files = os.listdir(images_source_folder)
    predict_class = conferenceType_class[conferenceType]
    for image_source_file in images_source_files:
        # if image_source_file in noiseImage_dict:
        #     continue
        if image_source_file not in needImage_dict:
            continue
        if image_source_file in complete_dict:
            count += 1
            continue
        imageName = image_source_file.split('.')[0]
        image_source_path = os.path.join(images_source_folder, image_source_file)
        results = {'0.3':None,}
        for confTHreshold, conf_folder in confThreshold_folder.items():
            result = model(image_source_path, classes=predict_class, conf=confTHreshold)[0]
            results[conf_folder] = result
        for conf_folder, result in results.items():
            result_box_conf_folder = os.path.join(result_box_folder, str(conf_folder), conferenceType)
            result_mask_conf_folder = os.path.join(result_mask_folder, str(conf_folder), conferenceType)
            result_conf_conf_folder = os.path.join(result_conf_folder, str(conf_folder), conferenceType)
            if not os.path.exists(result_box_conf_folder):
                os.makedirs(result_box_conf_folder)
            if not os.path.exists(result_mask_conf_folder):
                os.makedirs(result_mask_conf_folder)
            if not os.path.exists(result_conf_conf_folder):
                os.makedirs(result_conf_conf_folder)
            
            boxes = result.boxes
            masks = result.masks
            names = result.names
            pre_count = len(boxes.cls)
            if pre_count == 0:
                continue
            height, width = boxes.orig_shape
            confs = boxes.conf
            boxes_xyxyn = boxes.xyxyn # 多个box，每个box表示coco8seg形式的归一化坐标
            # masks_xyn = masks.xyn # 多个mask的array形式组成的列表，每个mask表示coco8seg形式的归一化坐标
            masks_xyn = []
            for mask in masks.data:
                if mask.max() == 0.0:
                    masks_xyn.append(None)
                    continue
                matrix = mask.cpu().numpy()
                matrix_boundary = np.array(get_boundary_indices(matrix), dtype=float)
                total_rows, total_cols = matrix.shape
                matrix_boundary[:,0] = matrix_boundary[:,0]/total_cols
                matrix_boundary[:,1] = matrix_boundary[:,1]/(((height/(width/total_cols)) + total_rows)/2)
                # row_indices, col_indices = np.where(matrix == 1.0)
                # row_ratios = row_indices / total_rows
                # col_ratios = col_indices / total_cols
                # mask_result = np.column_stack((col_ratios, row_ratios))
                masks_xyn.append(matrix_boundary)
                # rows_with_ones = np.unique(row_indices)
                # cols_with_ones = np.unique(col_indices)
            
            images_dist_file = os.path.join(forPredict_rootFolder, 'result', 'images', conf_folder, conferenceType, image_source_file)
            result_box_path = os.path.join(result_box_conf_folder, imageName + '.txt')
            result_mask_path = os.path.join(result_mask_conf_folder, imageName + '.txt')
            result_conf_path = os.path.join(result_conf_conf_folder, imageName + '.txt')
            result.save(images_dist_file)
            for conf, box_xyxyn, mask_xyn in zip(confs, boxes_xyxyn, masks_xyn):
                if mask_xyn is None:
                    continue
                with open(result_conf_path, 'a') as fConf, open(result_box_path, 'a') as fBox, open(result_mask_path, 'a') as fMask:
                    fConf.write(str(conf.item()) + '\n')
                    fBox.write(' '.join(box_xyxyn.cpu().numpy().astype(str)) + '\n')
                    fMask.write(' '.join(mask_xyn.reshape(1, -1)[0].astype(str)) + '\n')
        count += 1
        endTime = time.time()
        averTime = (endTime - startTime) / count
        print('*'*20)
        print(f"{conferenceType}\t{count}/{len(needImage_dict)}\taverTime:{round(averTime, 3)}")
        print('*'*20)