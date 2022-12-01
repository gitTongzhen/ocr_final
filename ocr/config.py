
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))


# 通用参数设置
r = 1               # 缩放比例
score_th = 0.8        # 置信度阈值，用于划分文字和图标
merge_box = False     # 是否合并文本检测框
use_mp = True         # 是否使用多线程
process_num = 10      # 线程数



# OCR参数

infer_h = 32
batch = 1
#----------------------------------- paddle系列 -------------------------------------------#
# pprec_v2.0 官方
model_path = os.path.join(cur_dir,"models/pprec_2.0.onnx")
in_names = "x"
out_names = ["softmax_0.tmp_0"]
keys_txt_path = os.path.join(cur_dir,"models/ppocr_keys_v1.txt")


# 文本图标检测
# 模型路径（pp-yolo-E）
# det_model_path =  os.path.join(cur_dir,"models/ppyoloe_crn_s_p2_alpha_80e_text_ico_1028.onnx")     # 推理尺寸640x640

# 文本检测onnx
# det_model_path = os.path.join(cur_dir,"models/ppyoloe_crn_s_p2_alpha_80e_text_small3_b1_wonms_sim.onnx")
# 文本检测openvino
# det_model_path = os.path.join(cur_dir,"models/ppyoloe_crn_s_p2_alpha_80e_text_small3_b1_wonms_sim.onnx")
det_model_path = os.path.join(cur_dir,"models/ppyoloe_crn_s_p2_alpha_80e_text_small3.onnx")
# 类别列表
label_path = os.path.join(cur_dir,"det/label_list_text_ico.txt")
# 显示置信度阈值
confThreshold = 0.4
# nms阈值
nmsThreshold = 0.6