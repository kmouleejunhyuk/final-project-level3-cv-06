import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
from pandas import json_normalize
from numba import njit
import cv2


def read_json(json_dir: str):
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    anno_info = json_normalize(source_anns['annotations'])
    image_info = json_normalize(source_anns['images'])
    return anno_info, image_info, source_anns


def edit_json(img_data, anno_data, categories, image_list):

    #지윤님 요청사항: 들어가는 부분만 처리해주세요
    if image_list:
        dummy = {
            "images":[],
            "categories": categories,
            "annotations": [],
        }
        image_dict = {}

        for k in image_list:
            image_dict[k] = 1

        imgid = {}
        idcount = 0
        for d in tqdm(img_data):
            if image_dict.get(d['file_name'], 0):
                p_id = d['id']
                d['id'] = idcount
                imgid[p_id] = idcount
                d['path'] = ''
                dummy['images'].append(d)
                idcount += 1

        idcount = 0
        for d in tqdm(anno_data):
            if d and imgid.get(d['image_id'], 0):
                d['id'] = idcount
                d['image_id'] = imgid[d['image_id']]
                dummy['annotations'].append(d)
                idcount += 1

        return dummy


def clean_json(data, image_list):
    delta = 0
    imgid = {}
    for i, d in enumerate(data['images']):
        imgid[d['id']] = d['id']
        if delta:
            p_id = d['id']
            d['id'] = d['id'] - delta
            imgid[p_id] = d['id']
        if d['file_name'] not in image_list:
            data['images'].pop(i)
            delta += 1

    delta = 0
    for i, d in enumerate(data['annotations']):
        d['id'] = d['id'] - delta
        if d['image_id'] not in list(imgid.keys()):
            data['annotations'].pop(i)
            delta += 1
        else:
            data['annotations'][i - delta]['image_id'] = imgid[d['image_id']]


@njit
def get_box(image: np.ndarray, bar = 75, pad = 20, side = 150):
    xmin, ymin, xmax, ymax = 1e+5, 1e+5, 0, 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pix = sum(image[i, j, :]) / 3

            if pix < 220:
                if i < ymin: 
                    ymin = i
                if j < xmin:
                    xmin = j
                if i > ymax:
                    ymax = i
                if j > xmax:
                    xmax = j
    
    if xmin - pad< 0:
        xmin = 0
    else:
        xmin = xmin - pad

    if ymin - pad< 0:
        ymin = 0
    else:
        ymin = ymin - pad

    return xmin,  ymin, xmax + pad, ymax + pad


def crop_resize_seg(Rx, Ry, x, y, xmin, ymin):
    x, y = x - xmin, y - ymin   #crop
    if x<0 or y<0:
         return None

    # return [round(Rx * x, 2), round(Ry * y, 2)]
    return [round(x, 2), round(y, 2)]


def crop_resize_bbox(Rx, Ry, box, xmin, ymin):
    x, y, w, h = box
    x, y = x - xmin, y - ymin   #crop
    if x<0 or y<0:
         return None
    # return [round(Rx * x, 2), round(Ry * y, 2), round(Rx * w, 2), round(Ry * h, 2)]
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def transform_all(image, bboxs, masks, target_size = 512):
    t_image = cv2.GaussianBlur(image, ksize=(7,7), sigmaX=0)
    xmin, ymin, xmax, ymax = map(int, get_box(t_image))
    Rx, Ry = target_size / (xmax - xmin), target_size / (ymax - ymin)

    r_mask, r_bbox = [], []

    #transform mask
    for key in masks.keys(): 
        for mask in masks[key]: #[[x, y, x1, y1 ...], [x, y, x1, y1 ...]]
            tmp = []
            assert len(mask) % 2 == 0
            while mask: 
                x = mask.pop(0)
                y = mask.pop(0)
                coord = crop_resize_seg(Rx, Ry, x, y, xmin, ymin)
                if coord:
                    x, y = coord
                    tmp.append(x)
                    tmp.append(y)  

        masks[key] = tmp 

    #transform box
    for key in bboxs.keys():
        box = bboxs[key]
        box = crop_resize_bbox(Rx, Ry, box, xmin, ymin)
        if box:
            bboxs[key] = box
        else:
            bboxs[key] = []

    #transform image
    image = image[ymin:ymax, xmin:xmax, :]
    # image = cv2.resize(image, dsize=(512, 512))

    return image, bboxs, masks


def df2dict(df):
    answer = {}
    for row in range(df.shape[0]):
        key = df.iloc[row, 0]
        val = df.iloc[row, 1]
        answer[key] = answer.get(key, []) + val
    return answer


def del_improper(imgdir, delete = True):
    try:
        img = cv2.imread(imgdir)
    except:
        if delete:
            os.remove(imgdir)
            return 0

    return img


if __name__ == '__main__':
    rootdir = r"D:\\resized\\train"
    dirlist = list(map(str, list(Path(rootdir).rglob('*.png'))))
    anno_info, image_info, origin = read_json(os.path.join(rootdir, 'modified_train_dummy.json'))
    
    img_list = []
    for imgdir in tqdm(dirlist):
        img = del_improper(imgdir)
        if not isinstance(img, np.ndarray):
            continue

        imgname = imgdir.split('\\')[-1]

        imgid = image_info.loc[image_info['file_name'] == imgname, 'id'].values[0]
        segs = anno_info.loc[anno_info['image_id'] == imgid, ['id', 'segmentation']]
        bboxs = anno_info.loc[anno_info['image_id'] == imgid, ['id', 'bbox']]

        segs = df2dict(segs)
        bboxs = df2dict(bboxs)

        img_list.append(imgname)
        cropped_img, cropped_bbox, cropped_mask = transform_all(img, bboxs, segs)

        #JHOH 요청: img width/height 처리
        image_index = image_info.loc[image_info['id'] == imgid].index.values[0]
        origin['images'][image_index]['height'] = cropped_img.shape[0]
        origin['images'][image_index]['width'] = cropped_img.shape[1]

        for k in cropped_mask.keys():
            idx, seg = k, cropped_mask[k]
            key = anno_info.loc[anno_info['id'] == idx].index.values[0]

            if seg:
                origin['annotations'][key]['segmentation'] = [seg]
            else:
                origin['annotations'][key] = {}

        for k in cropped_bbox.keys():
            idx, box = k, cropped_bbox[k]
            key = anno_info.loc[anno_info['id'] == idx].index.values[0]

            if box and origin['annotations'][key]:
                origin['annotations'][key]['bbox'] = box
            else:
                origin['annotations'][key] = {}

        cv2.imwrite(imgdir, cropped_img)

    origin = edit_json(origin['images'], origin['annotations'], origin['categories'], img_list)
    #쓰기
    print('start writing')
    with open(os.path.join(rootdir, 'modified_train_dummy_dummy.json'), 'w', encoding='UTF-8') as json_file:
        json.dump(origin, json_file)
    
    print('end')
