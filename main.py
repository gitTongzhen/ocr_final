

import cv2
import numpy as np
import os
from PIL import Image,ImageDraw,ImageFont
from ocr.pageItemRec import ocr_rec
import json
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
import time
rect_screen_num = []
all_cost = []
det_cost1 = []
# data_home = "../sumaitong/image1"
data_home = "./all"
# data_home = "D:\workspace\chinese_ocr\\test_version4\save1\save1"
# theard_pool = ThreadPoolExecutor(max_workers=2)
imgs = [img for img in os.listdir(data_home) if "mask" not in img]
for item in imgs:
    image_path = os.path.join(data_home,item)
    img = cv2.imread(image_path)[:,:,::-1]    # 直接转RGB 
    # rect_screen_shot.append(img)
    start_time = time.time()
    #result = ocr_rec(img)
    result = ocr_rec(img,use_mp=True,process_num=10)
    # result = all_result["texts"]
    rect_screen_num.append(len(result))
    # det_cost1.append(det_cost)
    # result = theard_pool.submit(ocr_rec,img)
    end_time = time.time()
    cost = end_time-start_time
    all_cost.append(cost)
    colors = ['red', 'green', 'blue', "purple"]
    img1 = Image.fromarray(img)
    img_detected = img1.copy()

    img_draw = ImageDraw.Draw(img_detected)
    for i, r in enumerate(result):
        rect_s,txt,score = r
        
        x1,y1,x2,y2 = np.array(rect_s).reshape(-1)
        size = max(min(x2-x1,y2-y1) // 2 , 20 )
        # fillcolor = colors[i % len(colors)]
        myfont = ImageFont.truetype(os.path.join(os.getcwd(), "./仿宋_GB2312.ttf"), size=size)
        img_draw.text((x1, y1 - size ), str(txt), font=myfont, fill=colors[i % len(colors)])
        img_draw.rectangle((x1,y1,x2+x1,y2+y1),outline=colors[i % len(colors)],width=2)

    print(type(img_draw))
    result_image_path = os.path.join(data_home+"_result",os.path.splitext(item)[0]+".png")
    img_detected.save(result_image_path)
    # print(result)
# ----------------------

print("real time:")
times = np.array(all_cost)
print(times)
print("avg time:", times.mean(axis=0))
print("num: ",np.mean(rect_screen_num))
# print("det cots: ",np.mean(det_cost1))