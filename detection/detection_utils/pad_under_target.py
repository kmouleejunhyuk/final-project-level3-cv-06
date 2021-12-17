#jiyoo seo
#delete annotation by image width/height

import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
from pandas import json_normalize
import pandas
from operator import add
from functools import reduce
import cv2
from numba import njit

def read_json(json_dir: str):
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    anno_info = json_normalize(source_anns['annotations'])
    image_info = json_normalize(source_anns['images'])
    return anno_info, image_info, source_anns

def del_improper(imgdir, delete = True):
    try:
#        print(imgdir)
        img = np.array(Image.open(imgdir))
    except:
        if delete:
            os.remove(imgdir)
            return 0

    return img

def create_new_json_wh(anno_info, image_info, origin, dirlist, target_size=(600,800)):
    img_list = []
    i =0 
    image_new = image_info.drop(image_info.index[0:len(image_info)])
    anno_new = anno_info.drop(anno_info.index[0:len(anno_info)])
    for imgdir in tqdm(dirlist):
        img = del_improper(imgdir)
        if not isinstance(img, np.ndarray):
            continue
        imgname = imgdir.split('/')[-1]
        if len(image_info.loc[image_info['file_name'] == imgname]) != 0:
            imgid, w, h = image_info.loc[image_info['file_name'] == imgname, ['id','width','height']].values[0]

            if target_size[0] >= h and target_size[1] >= w:
                if len(image_info.loc[image_info['id'] == imgid]) != 0:
                    image_tmp = image_info.loc[image_info['id'] == imgid]
                    anno_tmp = anno_info.loc[anno_info['image_id'] == imgid]
                    
                    image_new = image_new.append(image_tmp)
                    anno_new = anno_new.append(anno_tmp)
                    
        
    # df to json
    image_json = image_new.to_json(orient = 'table', index = False)    
    anno_json = anno_new.to_json(orient = 'table', index = False)    
    return image_json, anno_json
if __name__ == '__main__':
    rootdir = "/home/jiyouseo/Desktop/jupyter/final_project/train"
    dirlist = list(map(str, list(Path(rootdir).rglob('*png'))))
    anno_info, image_info, origin = read_json(os.path.join(rootdir, 'modified_train_dummy.json'))
    
    image_json, anno_json = create_new_json_wh(anno_info, image_info, origin, dirlist)
    
    tmp = origin.copy()
    tmp["images"] = eval(image_json)["data"]
    false = False
    tmp["annotations"] = eval(anno_json)["data"]

    print('start writing')
    with open(os.path.join(rootdir, 'create_new_json_wh.json'), 'w', encoding='UTF-8') as json_file:
        json.dump(tmp, json_file)

    print('end')
