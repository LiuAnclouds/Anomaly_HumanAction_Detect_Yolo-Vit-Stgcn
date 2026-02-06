import argparse
import cv2
import numpy as np
import torch
import time
import os
import json
import random
import yaml
import onnxruntime as ort
from collections import deque
from ultralytics import YOLO

# ---------------------------------------------------------------------
# 跌倒变量传出说明（bool / int）
# - config.processing.fall_confirm_frames: 连续多少帧预测为跌倒才判定为确认跌倒。
# - system.fall (bool): 是否有人确认跌倒；system.fall_consecutive (int): 最大连续跌倒帧数。
# - system.get_fall_status() -> {fall, fall_consecutive, by_track}。
# - run_main(config, fall_callback=lambda s: ...): 每帧调用 fall_callback(get_fall_status()) 传出。
# ---------------------------------------------------------------------


# =====================================================================
# 1. 核心工具函数 (预处理与后处理)
# =====================================================================

def preprocess_image(img_bgr, target_size=(192, 256)):
    """
    将 OpenCV 读取的 BGR 图片转换为 ViTPose 需要的 Tensor
    Output: Tensor (1, 3, 256, 192) Normalized, on GPU
    """
    img_resized = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    img_chw = img_normalized.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor.cuda()


def decode_keypoints(output_heatmap, original_box):
    """
    解析模型输出的热力图，映射回原图坐标
    返回: [[x, y, conf], ...] 共17个点
    """
    N, K, H, W = output_heatmap.shape
    heatmaps = output_heatmap.cpu().detach().numpy()[0]
    keypoints = []

    box_x1, box_y1, box_x2, box_y2 = original_box
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1

    for k in range(K):
        hm = heatmaps[k]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = idx
        conf = hm[y, x]

        scale_x = box_w / W
        scale_y = box_h / H

        final_x = int(box_x1 + x * scale_x)
        final_y = int(box_y1 + y * scale_y)

        keypoints.append([final_x, final_y, float(conf)])

    return keypoints


def self_norm(kpt, bbox):
    """
    归一化关键点坐标
    kpt: (2, T, 17, 1),  bbox: (T, 4) [x1, y1, w, h]
    返回: (2, T, 17, 1) 归一化后的坐标
    """
    # bbox格式: [x1, y1, w, h]
    tl = bbox[:, 0:2]  # 左上角坐标 (T, 2)
    wh = bbox[:, 2:]   # 宽高 (T, 2)
    
    # 扩展维度以匹配kpt的形状
    tl = np.expand_dims(np.transpose(tl, (1, 0)), (2, 3))  # (2, T, 1, 1)
    wh = np.expand_dims(np.transpose(wh, (1, 0)), (2, 3))   # (2, T, 1, 1)

    # 归一化: (kpt - tl) / wh，然后缩放到 (384, 512)
    res = (kpt - tl) / wh
    res *= np.expand_dims(np.array([[384.], [512.]]), (2, 3))
    return res


def convert_keypoints_to_stgcn_format(all_kpts, all_bbox, target_frames=50):
    """
    将关键点序列转换为STGCN需要的格式
    输入:
        all_kpts: list of list, 每帧的关键点 [[x, y, conf], ...] 共17个点
        all_bbox: list of list, 每帧的bbox [x1, y1, x2, y2]
        target_frames: 目标帧数，默认50
    返回:
        stgcn_input: (2, 50, 17, 1) numpy array
        scores: (50, 17, 1) numpy array
    """
    T = len(all_kpts)
    print(f"[转换] 输入帧数: {T}, 目标帧数: {target_frames}")
    
    if T == 0:
        print("[警告] 没有关键点数据，返回零数组")
        return np.zeros((2, target_frames, 17, 1), dtype=np.float32), \
               np.zeros((target_frames, 17, 1), dtype=np.float32)
    
    # 1. 转换为numpy数组
    # all_kpts: (T, 17, 3) -> 提取 (T, 17, 2) 坐标和 (T, 17, 1) 置信度
    kpts_array = np.array(all_kpts)  # (T, 17, 3)
    all_kpts_coords = kpts_array[:, :, :2]  # (T, 17, 2) [x, y]
    all_scores = kpts_array[:, :, 2:3]  # (T, 17, 1) [conf]
    
    # 2. 转换bbox格式: [x1, y1, x2, y2] -> [x1, y1, w, h]
    bbox_array = np.array(all_bbox)  # (T, 4)
    bbox_formatted = np.zeros((T, 4), dtype=np.float32)
    bbox_formatted[:, 0] = bbox_array[:, 0]  # x1
    bbox_formatted[:, 1] = bbox_array[:, 1]  # y1
    bbox_formatted[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]  # w = x2 - x1
    bbox_formatted[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]  # h = y2 - y1
    
    print(f"[转换] 关键点坐标范围: x=[{all_kpts_coords[:, :, 0].min():.1f}, {all_kpts_coords[:, :, 0].max():.1f}], "
          f"y=[{all_kpts_coords[:, :, 1].min():.1f}, {all_kpts_coords[:, :, 1].max():.1f}]")
    print(f"[转换] Bbox范围: w=[{bbox_formatted[:, 2].min():.1f}, {bbox_formatted[:, 2].max():.1f}], "
          f"h=[{bbox_formatted[:, 3].min():.1f}, {bbox_formatted[:, 3].max():.1f}]")
    
    # 3. 转置为 (2, T, 17, 1)
    keypoint = np.expand_dims(np.transpose(all_kpts_coords, [2, 0, 1]), -1)  # (2, T, 17, 1)
    print(f"[转换] 转置后形状: {keypoint.shape}")
    
    # 4. 归一化
    keypoint = self_norm(keypoint, bbox_formatted)
    print(f"[转换] 归一化后坐标范围: x=[{keypoint[0, :, :, 0].min():.2f}, {keypoint[0, :, :, 0].max():.2f}], "
          f"y=[{keypoint[1, :, :, 0].min():.2f}, {keypoint[1, :, :, 0].max():.2f}]")
    
    # 5. 统一帧数到target_frames
    # 策略：
    # - 超过target_frames帧: 连续截取中间target_frames帧
    # - 不足target_frames帧: 复制最后一帧进行补帧（而不是补0）
    # - 恰好target_frames帧: 无需处理
    current_frames = keypoint.shape[1]
    if current_frames > target_frames:
        # 超过target_frames帧: 连续截取中间target_frames帧
        frame_start = (current_frames - target_frames) // 2
        keypoint = keypoint[:, frame_start:frame_start + target_frames, :, :]
        all_scores = all_scores[frame_start:frame_start + target_frames, :, :]
        print(f"[转换] 超过{target_frames}帧（当前{current_frames}帧），连续截取中间{target_frames}帧（从第{frame_start}帧开始）")
    elif current_frames < target_frames:
        # 不足target_frames帧: 复制前面的帧进行补帧
        pad_length = target_frames - current_frames
        if current_frames > 0:
            # 策略：循环复制已有的帧，直到达到target_frames
            # 例如：如果有10帧，需要50帧，则复制为：1-10, 1-10, 1-10, 1-10, 1-10
            repeat_times = (pad_length // current_frames) + 1
            pad_kp_list = []
            pad_score_list = []
            
            # 复制整个序列多次
            for _ in range(repeat_times):
                pad_kp_list.append(keypoint)
                pad_score_list.append(all_scores)
            
            # 拼接并截取到需要的长度
            pad_kp = np.concatenate(pad_kp_list, axis=1)
            pad_score = np.concatenate(pad_score_list, axis=0)
            
            # 只取前面需要的部分
            pad_kp = pad_kp[:, :pad_length, :, :]
            pad_score = pad_score[:pad_length, :, :]
            
            keypoint = np.concatenate([keypoint, pad_kp], axis=1)
            all_scores = np.concatenate([all_scores, pad_score], axis=0)
            print(f"[转换] 不足{target_frames}帧（当前{current_frames}帧），循环复制前面的帧补到{target_frames}帧")
        else:
            # 如果没有帧，补0
            keypoint = np.concatenate([
                keypoint,
                np.zeros((2, pad_length, 17, 1), dtype=keypoint.dtype)
            ], axis=1)
            all_scores = np.concatenate([
                all_scores,
                np.zeros((pad_length, 17, 1), dtype=all_scores.dtype)
            ], axis=0)
            print(f"[转换] 没有关键点数据，补0到{target_frames}帧")
    else:
        # 恰好target_frames帧: 无需处理
        print(f"[转换] 恰好{target_frames}帧，无需处理")
    
    # 确保最终形状正确
    assert keypoint.shape == (2, target_frames, 17, 1), \
        f"关键点形状错误: {keypoint.shape}, 期望: (2, {target_frames}, 17, 1)"
    assert all_scores.shape == (target_frames, 17, 1), \
        f"置信度形状错误: {all_scores.shape}, 期望: ({target_frames}, 17, 1)"
    
    print(f"[转换] 最终形状: keypoint={keypoint.shape}, scores={all_scores.shape}")
    print(f"[转换] 最终坐标范围: x=[{keypoint[0, :, :, 0].min():.2f}, {keypoint[0, :, :, 0].max():.2f}], "
          f"y=[{keypoint[1, :, :, 0].min():.2f}, {keypoint[1, :, :, 0].max():.2f}]")
    
    return keypoint, all_scores


def draw_skeleton(img, keypoints, dataset_name='coco', conf_thres=0.4, config=None, keypoint_names=None):
    """
    绘制骨架 + 显示关键点名称和置信度
    支持 COCO (17点) 和 AIC (14点)
    """
    # 使用配置或默认值
    if config is None:
        draw_config = {
            'keypoint': {'enabled': True, 'color': [0, 255, 0], 'radius': 4, 'show_conf': True, 
                        'show_name': False, 'conf_threshold': 0.4, 'name_offset': [0, -15], 'conf_offset': [0, -5]},
            'skeleton': {'enabled': True, 'color': [255, 200, 0], 'thickness': 2}
        }
    else:
        draw_config = config.get('drawing', {})
    
    # 获取关键点配置和阈值
    kp_config = draw_config.get('keypoint', {})
    # 如果配置中有conf_threshold，优先使用配置值
    if 'conf_threshold' in kp_config:
        conf_thres = kp_config['conf_threshold']
    kp_conf_thres = conf_thres  # 统一使用这个阈值
    
    skeleton_links = []
    if dataset_name == 'coco':
        skeleton_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    elif dataset_name == 'aic':
        skeleton_links = [
            (12, 13), (13, 0), (13, 3),
            (0, 1), (1, 2), (3, 4), (4, 5),
            (0, 6), (3, 9), (6, 7), (7, 8), (9, 10), (10, 11)
        ]
    else:
        print(f"[警告] 未知数据集: {dataset_name}, 跳过骨架连线")

    # 绘制关键点
    if kp_config.get('enabled', True):
        kp_color = tuple(kp_config.get('color', [0, 255, 0]))
        kp_radius = kp_config.get('radius', 4)
        show_conf = kp_config.get('show_conf', True)
        show_name = kp_config.get('show_name', False)
        name_offset = kp_config.get('name_offset', [0, -15])
        conf_offset = kp_config.get('conf_offset', [0, -5])
        
        for idx, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > kp_conf_thres:
                # 绘制关键点
                cv2.circle(img, (int(x), int(y)), kp_radius, kp_color, -1)
                
                # 显示关键点名称和置信度
                if show_name and keypoint_names and idx in keypoint_names:
                    kp_name = keypoint_names[idx]
                    name_x = int(x) + name_offset[0]
                    name_y = int(y) + name_offset[1]
                    # 如果显示名称和置信度，将它们组合在一起
                    if show_conf:
                        label = f"{kp_name}:{conf:.2f}"
                    else:
                        label = kp_name
                    cv2.putText(img, label, (name_x, name_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                elif show_conf:
                    # 只显示置信度
                    conf_x = int(x) + conf_offset[0]
                    conf_y = int(y) + conf_offset[1]
                    conf_label = f"{conf:.2f}"
                    cv2.putText(img, conf_label, (conf_x, conf_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 绘制骨架连线
    skel_config = draw_config.get('skeleton', {})
    if skel_config.get('enabled', True):
        skel_color = tuple(skel_config.get('color', [255, 200, 0]))
        skel_thickness = skel_config.get('thickness', 2)
        
        for idx_a, idx_b in skeleton_links:
            if idx_a < len(keypoints) and idx_b < len(keypoints):
                kp_a = keypoints[idx_a]
                kp_b = keypoints[idx_b]
                if kp_a[2] > kp_conf_thres and kp_b[2] > kp_conf_thres:
                    cv2.line(img, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])),
                             skel_color, skel_thickness)
    return img


# =====================================================================
# 2. 主类: 跌倒检测系统
# =====================================================================

class FallDetectionSystem:
    def __init__(self, vitpose_path, yolo_path, stgcn_path, device='cuda', window_size=50, 
                 stgcn_provider='CPUExecutionProvider', config=None):
        self.device = torch.device(device)
        self.window_size = window_size
        self.config = config or {}
        
        # ID映射配置
        id_config = self.config.get('id_mapping', {})
        self.id_mapping_enabled = id_config.get('enabled', False)
        self.id_mapping_auto = id_config.get('auto_generate', True)
        self.id_mapping = id_config.get('mapping', {})
        
        # 类别映射配置
        class_mapping = self.config.get('class_mapping', {})
        self.class_names = {
            0: class_mapping.get('0', class_mapping.get(0, 'Fall')),  # 支持字符串键和整数键
            1: class_mapping.get('1', class_mapping.get(1, 'Normal'))
        }
        
        # COCO关键点名称映射
        self.coco_keypoint_names = {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }
        
        # 缓冲区倍数
        buffer_multiplier = self.config.get('processing', {}).get('buffer_multiplier', 2)
        
        # 二次确认：连续多少帧预测为跌倒才判定为跌倒
        self.fall_confirm_frames = max(1, int(self.config.get('processing', {}).get('fall_confirm_frames', 3)))
        self.fall_consecutive_count = {}  # {track_id: int} 连续跌倒预测次数
        
        # 存储每个track_id的STGCN结果（用于在检测框旁边显示）
        self.stgcn_results = {}  # {track_id: {'class': int, 'prob': float, 'conf': float, 'fall_confirmed': bool, 'fall_consecutive_count': int}}
        
        # 对外暴露的跌倒变量（bool / int），供外部读取
        self.fall = False           # bool: 是否有人确认跌倒
        self.fall_consecutive = 0   # int: 当前最大连续跌倒帧数（或主目标）
        
        print(f"[初始化] 加载YOLO: {yolo_path}")
        self.yolo = YOLO(yolo_path)

        print(f"[初始化] 加载ViTPose: {vitpose_path}")
        if not os.path.exists(vitpose_path):
            raise FileNotFoundError(f"文件不存在: {vitpose_path}")
        with open(vitpose_path, 'rb') as f:
            self.vitpose = torch.jit.load(f, map_location=self.device)
        self.vitpose.eval()

        print(f"[初始化] 加载STGCN: {stgcn_path}")
        if not os.path.exists(stgcn_path):
            raise FileNotFoundError(f"文件不存在: {stgcn_path}")
        # 根据配置选择provider
        providers = [stgcn_provider]
        if stgcn_provider == 'CUDAExecutionProvider':
            providers.append('CPUExecutionProvider')  # 添加CPU作为后备
        self.stgcn_session = ort.InferenceSession(stgcn_path, providers=providers)
        self.stgcn_input_name = self.stgcn_session.get_inputs()[0].name
        self.stgcn_output_name = self.stgcn_session.get_outputs()[0].name
        print(f"[初始化] STGCN输入名称: {self.stgcn_input_name}, 输出名称: {self.stgcn_output_name}")
        
        # 检查STGCN输入形状
        input_shape = self.stgcn_session.get_inputs()[0].shape
        print(f"[初始化] STGCN期望输入形状: {input_shape}")

        print("[初始化] 预热模型...")
        dummy = torch.randn(1, 3, 256, 192).cuda()
        with torch.no_grad():
            self.vitpose(dummy)
        print(f"[初始化] 二次确认帧数: {self.fall_confirm_frames}（连续>=此值才判定跌倒）")
        print("[初始化] 系统就绪")

        # 关键点序列缓冲区（按track_id存储）
        self.keypoint_buffers = {}  # {track_id: {'kpts': deque, 'bboxes': deque}}
        self.track_id_counter = 0
        self.buffer_maxlen = window_size * buffer_multiplier
    
    def get_display_id(self, track_id):
        """获取显示用的ID（支持自动生成Person1/Person2或手动映射）"""
        if not self.id_mapping_enabled:
            return f"ID{track_id}"
        
        if self.id_mapping_auto:
            # 自动生成Person1, Person2等
            return f"Person{track_id + 1}"
        else:
            # 使用手动映射
            if track_id in self.id_mapping:
                return self.id_mapping[track_id]
            return f"Person{track_id + 1}"  # 如果没有映射，也使用自动生成

    def _update_fall_status(self):
        """根据 stgcn_results 更新对外暴露的 fall / fall_consecutive"""
        any_confirmed = False
        max_consecutive = 0
        for r in self.stgcn_results.values():
            if r.get('fall_confirmed', False):
                any_confirmed = True
            n = r.get('fall_consecutive_count', 0)
            if n > max_consecutive:
                max_consecutive = n
        self.fall = any_confirmed
        self.fall_consecutive = max_consecutive

    def get_fall_status(self):
        """
        获取跌倒状态，供外部读取。
        返回:
            fall: bool, 是否有人确认跌倒
            fall_consecutive: int, 当前最大连续跌倒帧数
            by_track: {track_id: {'fall': bool, 'fall_consecutive': int}}
        """
        by_track = {}
        for tid, r in self.stgcn_results.items():
            by_track[tid] = {
                'fall': r.get('fall_confirmed', False),
                'fall_consecutive': r.get('fall_consecutive_count', 0)
            }
        return {
            'fall': self.fall,
            'fall_consecutive': self.fall_consecutive,
            'by_track': by_track
        }

    def process_frame(self, frame, dataset_name='coco', box_conf_thres=0.5, 
                     pose_conf_thres=0.4, run_pose_estimation=True):
        """
        处理单帧
        返回: frame, all_keypoints, detection_results
        """
        img_h, img_w = frame.shape[:2]
        
        # 获取绘制配置
        draw_config = self.config.get('drawing', {})
        bbox_config = draw_config.get('bbox', {})
        text_config = draw_config.get('text', {})
        min_box_size = self.config.get('detection', {}).get('min_box_size', 10)

        # 1. YOLO检测
        results = self.yolo.predict(frame, conf=box_conf_thres, classes=0, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # 按置信度排序，并限制最多处理的人数
        max_persons = self.config.get('detection', {}).get('max_persons', 1)
        order = np.argsort(-confs)  # 从高到低
        if max_persons > 0:
            order = order[:max_persons]
        boxes = boxes[order]
        confs = confs[order]

        all_keypoints = []
        current_track_ids = []

        for i, box in enumerate(boxes):
            box_score = confs[i]
            if box_score < box_conf_thres:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                continue

            # 简单的track_id分配（实际应该用MOT）
            track_id = i  # 简化处理，实际应该用跟踪算法
            display_id = self.get_display_id(track_id)

            # 绘制YOLO框
            if bbox_config.get('enabled', True):
                # 仅当二次确认跌倒（fall_confirmed）时，检测框与文字高亮为红色
                is_fall = False
                fall_color = tuple(self.config.get('display', {}).get('result', {}).get('fall_color', [0, 0, 255]))
                normal_color = tuple(self.config.get('display', {}).get('result', {}).get('normal_color', [0, 255, 0]))
                if track_id in self.stgcn_results:
                    try:
                        is_fall = bool(self.stgcn_results[track_id].get('fall_confirmed', False))
                    except Exception:
                        is_fall = False

                bbox_color = fall_color if is_fall else tuple(bbox_config.get('color', [255, 255, 0]))
                bbox_thickness = bbox_config.get('thickness', 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
                
                # 显示Person ID和检测框置信度
                text_color = fall_color if is_fall else tuple(text_config.get('color', [255, 255, 0]))
                text_scale = text_config.get('scale', 0.5)
                text_thickness = text_config.get('thickness', 2)
                font = getattr(cv2, text_config.get('font', 'FONT_HERSHEY_SIMPLEX'))
                
                label_y = y1 - 10
                # Person ID
                cv2.putText(frame, display_id, (x1, label_y),
                            font, text_scale, text_color, text_thickness)
                
                # 检测框置信度
                if bbox_config.get('show_score', True):
                    box_label = f"Box:{box_score:.2f}"
                    cv2.putText(frame, box_label, (x1, label_y + 20),
                                font, text_scale * 0.8, text_color, text_thickness)
                
                # 在检测框旁边显示STGCN结果（如果有）
                if bbox_config.get('show_stgcn_result', True) and track_id in self.stgcn_results:
                    stgcn_result = self.stgcn_results[track_id]
                    class_id = stgcn_result['class']
                    class_name = self.class_names.get(class_id, f"Class{class_id}")
                    prob = stgcn_result.get('prob', stgcn_result.get('fall_prob', 0.0))
                    conf = stgcn_result.get('conf', prob)  # conf使用概率值（百分制）
                    n_consec = stgcn_result.get('fall_consecutive_count', 0)
                    confirmed = stgcn_result.get('fall_confirmed', False)
                    
                    # 根据二次确认结果选择颜色（确认跌倒才标红）
                    if confirmed:
                        result_color = fall_color
                    else:
                        result_color = normal_color
                    
                    offset = bbox_config.get('stgcn_result_offset', [5, 25])
                    result_x = x2 + offset[0]
                    result_y = y1 + offset[1]
                    
                    # 显示类别、概率、置信度、连续跌倒帧数：分多行绘制（cv2.putText 不支持 \n）
                    line_gap = max(14, int(20 * text_scale))
                    cv2.putText(frame, f"{class_name}", (result_x, result_y),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(frame, f"Prob:{prob*100:.1f}%", (result_x, result_y + line_gap),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(frame, f"Conf:{conf*100:.1f}%", (result_x, result_y + 2 * line_gap),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(frame, f"Fall:{n_consec}/{self.fall_confirm_frames}" + (" OK" if confirmed else ""),
                                (result_x, result_y + 3 * line_gap), font, text_scale, result_color, text_thickness)

            # 姿态估计
            if run_pose_estimation:
                person_crop = frame[y1:y2, x1:x2]
                input_tensor = preprocess_image(person_crop, target_size=(192, 256))

                with torch.no_grad():
                    output = self.vitpose(input_tensor)

                kpts = decode_keypoints(output, [x1, y1, x2, y2])
                all_keypoints.append(kpts)

                # 存储到缓冲区
                if track_id not in self.keypoint_buffers:
                    self.keypoint_buffers[track_id] = {
                        'kpts': deque(maxlen=self.buffer_maxlen),
                        'bboxes': deque(maxlen=self.buffer_maxlen)
                    }
                
                self.keypoint_buffers[track_id]['kpts'].append(kpts)
                self.keypoint_buffers[track_id]['bboxes'].append([x1, y1, x2, y2])
                current_track_ids.append(track_id)

                # 绘制骨架（传入关键点名称）
                keypoint_names = self.coco_keypoint_names if dataset_name == 'coco' else None
                draw_skeleton(frame, kpts, dataset_name=dataset_name, 
                            conf_thres=pose_conf_thres, config=self.config,
                            keypoint_names=keypoint_names)

        # 清理丢失的track_id，并重置其连续跌倒计数
        lost_ids = set(self.keypoint_buffers.keys()) - set(current_track_ids)
        for lost_id in lost_ids:
            if lost_id in self.keypoint_buffers:
                del self.keypoint_buffers[lost_id]
            if lost_id in self.stgcn_results:
                del self.stgcn_results[lost_id]
            if lost_id in self.fall_consecutive_count:
                del self.fall_consecutive_count[lost_id]
        if lost_ids:
            self._update_fall_status()

        return frame, all_keypoints, current_track_ids

    def predict_fall(self, track_id):
        """
        对指定track_id进行跌倒检测
        返回: {'class': 0或1, 'score': float, 'fall_prob': float}
        """
        if track_id not in self.keypoint_buffers:
            return None
        
        buffer = self.keypoint_buffers[track_id]
        kpts_list = list(buffer['kpts'])
        bboxes_list = list(buffer['bboxes'])
        
        # 获取最小预测帧数配置
        min_frames = self.config.get('processing', {}).get('min_frames_for_prediction', self.window_size)
        use_sliding = self.config.get('processing', {}).get('use_sliding_window', True)

        # 检查是否有足够的数据
        if len(kpts_list) < min_frames:
            return None  # 数据不足，不进行预测

        # 使用滑动窗口：只使用最新的window_size帧（确保实时性）
        if use_sliding and len(kpts_list) > self.window_size:
            kpts_list = kpts_list[-self.window_size:]  # 只取最新的window_size帧
            bboxes_list = bboxes_list[-self.window_size:]
            print(f"\n[STGCN推理] Track ID: {track_id}, 缓冲区总帧数: {len(buffer['kpts'])}, 使用最新{self.window_size}帧（滑动窗口，实时更新）")
        elif use_sliding:
            # 即使不足window_size，也使用所有可用帧（会在convert_keypoints_to_stgcn_format中补帧）
            print(f"\n[STGCN推理] Track ID: {track_id}, 帧数: {len(kpts_list)}（不足{self.window_size}帧，将补帧）")
        else:
            print(f"\n[STGCN推理] Track ID: {track_id}, 帧数: {len(kpts_list)}（使用全部缓冲区）")

        # 转换为STGCN格式
        stgcn_input, scores = convert_keypoints_to_stgcn_format(
            kpts_list, bboxes_list, target_frames=self.window_size
        )

        # 添加batch维度: (2, 50, 17, 1) -> (1, 2, 50, 17, 1)
        stgcn_input_batch = np.expand_dims(stgcn_input, axis=0).astype(np.float32)
        print(f"[STGCN推理] 输入形状: {stgcn_input_batch.shape}")
        print(f"[STGCN推理] 输入数据范围: min={stgcn_input_batch.min():.4f}, max={stgcn_input_batch.max():.4f}, "
              f"mean={stgcn_input_batch.mean():.4f}")

        # STGCN推理
        try:
            outputs = self.stgcn_session.run(
                [self.stgcn_output_name],
                {self.stgcn_input_name: stgcn_input_batch}
            )
            output_logit = outputs[0][0]  # (num_classes,)
            print(f"[STGCN推理] 输出logits: {output_logit}")

            # 后处理
            # STGCN输出的是logits，需要先计算softmax得到概率
            predicted_class = int(np.argmax(output_logit))  # 使用argmax找到预测类别

            # 计算softmax概率（将logits转换为0-1之间的概率）
            exp_logits = np.exp(output_logit - np.max(output_logit))  # 数值稳定
            probs = exp_logits / np.sum(exp_logits)

            # 获取预测类别的概率（0-1之间，百分制）
            class_prob = float(probs[predicted_class])
            fall_prob = float(probs[0])  # 跌倒概率
            normal_prob = float(probs[1]) if len(probs) > 1 else 0.0  # 正常概率

            # conf使用softmax后的概率值（百分制），而不是logit值
            conf = class_prob

            # 二次确认：连续 N 帧预测为跌倒才判定为确认跌倒
            if predicted_class == 0:  # Fall
                self.fall_consecutive_count[track_id] = self.fall_consecutive_count.get(track_id, 0) + 1
            else:  # Normal
                self.fall_consecutive_count[track_id] = 0
            n_consecutive = self.fall_consecutive_count[track_id]
            fall_confirmed = n_consecutive >= self.fall_confirm_frames

            result = {
                'class': predicted_class,
                'score': float(output_logit[predicted_class]),  # 原始logit值（用于调试）
                'prob': class_prob,  # 预测类别的概率（0-1）
                'conf': conf,  # 置信度（使用概率值，百分制）
                'fall_prob': fall_prob,  # 跌倒概率
                'normal_prob': normal_prob,  # 正常概率
                'all_logits': output_logit.tolist(),
                'all_probs': probs.tolist(),
                'fall_confirmed': fall_confirmed,       # bool: 是否确认跌倒
                'fall_consecutive_count': n_consecutive # int: 连续跌倒帧数
            }

            # 存储结果以便在检测框旁边显示
            self.stgcn_results[track_id] = {
                'class': predicted_class,
                'prob': class_prob,
                'conf': conf,  # 使用概率值作为置信度
                'fall_prob': fall_prob,
                'normal_prob': normal_prob,
                'fall_confirmed': fall_confirmed,
                'fall_consecutive_count': n_consecutive
            }

            # 更新对外暴露的跌倒变量（bool / int）
            self._update_fall_status()

            class_name = self.class_names.get(predicted_class, f"Class{predicted_class}")
            print(f"[STGCN推理] 预测结果: class={predicted_class} ({class_name}), "
                  f"prob={class_prob:.4f} ({class_prob*100:.2f}%), conf={conf:.4f} ({conf*100:.2f}%), "
                  f"fall_prob={fall_prob:.4f} ({fall_prob*100:.2f}%), "
                  f"连续跌倒={n_consecutive}/{self.fall_confirm_frames}, 确认跌倒={fall_confirmed}")
            return result
        except Exception as e:
            print(f"[STGCN推理] 错误: {e}")
            import traceback
            traceback.print_exc()
            return None


# =====================================================================
# 3. 图片预测模式
# =====================================================================

def predict_image(config):
    """
    图片预测模式：处理单张图片
    """
    device = 'cuda' if (config['device']['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    system = FallDetectionSystem(
        config['models']['vitpose'],
        config['models']['yolo'],
        config['models']['stgcn'],
        device=device,
        window_size=config['processing']['window_size'],
        stgcn_provider=config['device']['stgcn_provider'],
        config=config
    )

    # 读取图片
    input_path = config['input']['source']
    if not os.path.exists(input_path):
        print(f"[错误] 图片文件不存在: {input_path}")
        return

    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[错误] 无法读取图片: {input_path}")
        return

    print(f"\n[图片预测] 图片路径: {input_path}")
    print(f"[图片预测] 图片尺寸: {frame.shape[1]}x{frame.shape[0]}")
    print(f"[图片预测] 数据集: {config['detection']['dataset'].upper()}")
    print(f"[图片预测] Box阈值: {config['detection']['box_conf_threshold']}, Pose阈值: {config['detection']['pose_conf_threshold']}\n")

    # 旋转处理
    rotate = config['input']['rotate']
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # 处理图片
    res_img, current_kpts, track_ids = system.process_frame(
        frame.copy(),
        dataset_name=config['detection']['dataset'],
        box_conf_thres=config['detection']['box_conf_threshold'],
        pose_conf_thres=config['detection']['pose_conf_threshold'],
        run_pose_estimation=True
    )

    print(f"[图片预测] 检测到 {len(track_ids)} 个人")

    # 对每个检测到的人进行跌倒检测
    # 由于只有单张图片，需要将关键点重复50次来满足STGCN的输入要求
    results = []
    for track_id in track_ids:
        if track_id in system.keypoint_buffers:
            buffer = system.keypoint_buffers[track_id]
            kpts_list = list(buffer['kpts'])
            bboxes_list = list(buffer['bboxes'])

            if len(kpts_list) == 0:
                continue

            window_size = config['processing']['window_size']
            print(f"\n[图片预测] 处理 Track ID: {track_id}")
            print(f"[图片预测] 关键点帧数: {len(kpts_list)} (单张图片，将重复到{window_size}帧)")

            # 将单帧关键点重复到window_size帧
            kpts_repeated = kpts_list * window_size
            bboxes_repeated = bboxes_list * window_size

            # 转换为STGCN格式
            stgcn_input, scores = convert_keypoints_to_stgcn_format(
                kpts_repeated[:window_size],
                bboxes_repeated[:window_size],
                target_frames=window_size
            )

            # 添加batch维度
            stgcn_input_batch = np.expand_dims(stgcn_input, axis=0).astype(np.float32)
            print(f"[图片预测] STGCN输入形状: {stgcn_input_batch.shape}")
            print(f"[图片预测] STGCN输入数据范围: min={stgcn_input_batch.min():.4f}, max={stgcn_input_batch.max():.4f}, "
                  f"mean={stgcn_input_batch.mean():.4f}")

            # STGCN推理
            try:
                outputs = system.stgcn_session.run(
                    [system.stgcn_output_name],
                    {system.stgcn_input_name: stgcn_input_batch}
                )
                output_logit = outputs[0][0]
                print(f"[图片预测] STGCN输出logits: {output_logit}")

                # 后处理
                predicted_class = int(np.argmax(output_logit))  # 使用argmax找到预测类别

                # 计算softmax概率（将logits转换为0-1之间的概率）
                exp_logits = np.exp(output_logit - np.max(output_logit))
                probs = exp_logits / np.sum(exp_logits)

                class_prob = float(probs[predicted_class])
                fall_prob = float(probs[0])
                normal_prob = float(probs[1]) if len(probs) > 1 else 0.0

                # conf使用softmax后的概率值（百分制），而不是logit值
                conf = class_prob

                # 图片模式无多帧，不二次确认；直接按原始预测设置
                fall_confirmed_img = (predicted_class == 0)
                fall_consecutive_img = 1 if predicted_class == 0 else 0

                result = {
                    'track_id': track_id,
                    'class': predicted_class,
                    'score': float(output_logit[predicted_class]),  # 原始logit值（用于调试）
                    'prob': class_prob,
                    'conf': conf,  # 置信度（使用概率值，百分制）
                    'fall_prob': fall_prob,
                    'normal_prob': normal_prob,
                    'all_logits': output_logit.tolist(),
                    'all_probs': probs.tolist(),
                    'fall_confirmed': fall_confirmed_img,
                    'fall_consecutive_count': fall_consecutive_img
                }
                results.append(result)

                # 存储结果以便在检测框旁边显示
                system.stgcn_results[track_id] = {
                    'class': predicted_class,
                    'prob': class_prob,
                    'conf': conf,  # 使用概率值作为置信度
                    'fall_prob': fall_prob,
                    'normal_prob': normal_prob,
                    'fall_confirmed': fall_confirmed_img,
                    'fall_consecutive_count': fall_consecutive_img
                }
                system._update_fall_status()

                display_id = system.get_display_id(track_id)
                class_name = system.class_names.get(predicted_class, f"Class{predicted_class}")
                print(f"[图片预测] {display_id}: {class_name} "
                      f"prob={class_prob:.4f} ({class_prob*100:.2f}%), conf={conf:.4f} ({conf*100:.2f}%), "
                      f"fall_prob={fall_prob:.4f} ({fall_prob*100:.2f}%)")

                # 重新处理图片以显示STGCN结果在检测框旁边
                res_img, _, _ = system.process_frame(
                    frame.copy(),
                    dataset_name=config['detection']['dataset'],
                    box_conf_thres=config['detection']['box_conf_threshold'],
                    pose_conf_thres=config['detection']['pose_conf_threshold'],
                    run_pose_estimation=True
                )

            except Exception as e:
                print(f"[图片预测] STGCN推理错误: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果图片
    if config['output']['image_path']:
        output_path = config['output']['image_path']
    else:
        # 自动生成输出路径
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else '.'
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

    cv2.imwrite(output_path, res_img)
    print(f"\n[图片预测] 结果已保存到: {output_path}")

    # 显示结果（可选）
    if config['output']['show_result']:
        cv2.imshow("Fall Detection Result", res_img)
        print("[图片预测] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def run_main(config, fall_callback=None):
    """
    视频/摄像头模式。
    fall_callback: 可选，每帧调用 callback(get_fall_status()) 传出跌倒变量，便于外部告警等。
    """
    device = 'cuda' if (config['device']['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    system = FallDetectionSystem(
        config['models']['vitpose'],
        config['models']['yolo'],
        config['models']['stgcn'],
        device=device,
        window_size=config['processing']['window_size'],
        stgcn_provider=config['device']['stgcn_provider'],
        config=config
    )

    # 打开视频源
    input_source = config['input']['source']
    print(f"[视频源] 正在打开: {input_source}")
    
    # 为了避免某些后端阻塞，这里在 Windows 上优先使用 DirectShow
    if input_source.isdigit():
        cam_id = int(input_source)
        print(f"[视频源] 尝试通过 DirectShow 打开摄像头 ID: {cam_id}")
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        # 如果 DirectShow 打不开，再退回默认后端
        if not cap.isOpened():
            print("[视频源] DirectShow 打开失败，尝试默认后端...")
            cap.release()
            cap = cv2.VideoCapture(cam_id)
    else:
        # 文件/网络流一般不会阻塞，仍用默认后端
        cap = cv2.VideoCapture(input_source)
        print(f"[视频源] 尝试打开视频文件: {input_source}")

    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {input_source}")
        print(f"[错误] 请检查:")
        print(f"  - 摄像头是否已连接并正常工作")
        print(f"  - 摄像头ID是否正确（尝试 0 或 1）")
        print(f"  - 视频文件路径是否正确")
        return

    print(f"[视频源] 视频源已打开，正在获取属性...")
    
    # 设置一些摄像头属性（避免阻塞）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区，提高实时性
    
    # 尝试读取第一帧来验证摄像头是否正常工作
    print(f"[视频源] 尝试读取第一帧...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"[错误] 无法从视频源读取帧")
        print(f"[错误] 摄像头可能被其他程序占用或无法正常工作")
        cap.release()
        return
    
    print(f"[视频源] 成功读取第一帧，尺寸: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # 获取视频属性（使用实际读取的帧尺寸，避免阻塞）
    width = test_frame.shape[1]
    height = test_frame.shape[0]
    
    # 尝试获取FPS（如果失败则使用默认值）
    try:
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0:
            fps_video = 30
            print(f"[视频源] 无法获取FPS，使用默认值: {fps_video}")
    except:
        fps_video = 30
        print(f"[视频源] 获取FPS失败，使用默认值: {fps_video}")

    print(f"\n[开始推理] 分辨率: {width}x{height}, FPS: {fps_video}")
    print(f"[开始推理] 数据集: {config['detection']['dataset'].upper()}")
    print(f"[开始推理] Box阈值: {config['detection']['box_conf_threshold']}, Pose阈值: {config['detection']['pose_conf_threshold']}")
    print(f"[开始推理] 窗口大小: {config['processing']['window_size']}帧")
    print(f"[开始推理] 按 'q' 退出, 'p' 打印调试信息\n")

    frame_count = 0
    last_prediction_frame = {}
    
    print(f"[开始推理] 进入主循环，开始处理帧...\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"[警告] 无法读取帧（可能视频结束或摄像头断开），退出")
            break
        
        if frame is None or frame.size == 0:
            print(f"[警告] 读取到空帧，跳过")
            continue

        frame_count += 1

        rotate = config['input']['rotate']
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 跳帧逻辑
        pose_interval = config['processing']['pose_interval']
        run_pose = (frame_count % pose_interval == 0)

        # 处理帧
        res_img, current_kpts, track_ids = system.process_frame(
            frame,
            dataset_name=config['detection']['dataset'],
            box_conf_thres=config['detection']['box_conf_threshold'],
            pose_conf_thres=config['detection']['pose_conf_threshold'],
            run_pose_estimation=run_pose
        )

        # 对每个track_id进行跌倒检测（当数据足够时）
        window_size = config['processing']['window_size']
        prediction_interval = config['processing']['prediction_interval']
        min_frames = config['processing'].get('min_frames_for_prediction', window_size)

        for track_id in track_ids:
            if track_id in system.keypoint_buffers:
                buffer = system.keypoint_buffers[track_id]
                buffer_len = len(buffer['kpts'])

                # 检查是否有足够的数据进行预测
                if buffer_len >= min_frames:
                    # 每N帧预测一次（避免频繁预测）
                    if track_id not in last_prediction_frame or \
                       (frame_count - last_prediction_frame[track_id]) >= prediction_interval:
                        result = system.predict_fall(track_id)
                        if result:
                            last_prediction_frame[track_id] = frame_count
                            # STGCN结果已在 process_frame 中显示；fall/fall_consecutive 已更新
                else:
                    # 显示等待状态
                    if buffer_len > 0:
                        display_id = system.get_display_id(track_id)
                        progress = (buffer_len / window_size) * 100
                        # 可选：在检测框旁边显示收集进度
                        pass  # 暂时不显示，避免界面混乱

        fps_real = 1.0 / (time.time() - t0 + 1e-5)

        # UI显示（使用配置）
        ui_config = config.get('display', {}).get('ui', {})
        window_name = config.get('output', {}).get('window_name', 'Fall Detection')

        if ui_config.get('enabled', True):
            # FPS显示
            fps_config = ui_config.get('fps', {})
            if fps_config.get('enabled', True):
                pos = fps_config.get('position', [20, 40])
                color = tuple(fps_config.get('color', [0, 255, 0]))
                cv2.putText(res_img, f"FPS: {fps_real:.1f}", tuple(pos),
                           cv2.FONT_HERSHEY_SIMPLEX, fps_config.get('font_scale', 1.0),
                           color, fps_config.get('thickness', 2))

            # 帧数显示
            frame_config = ui_config.get('frame_count', {})
            if frame_config.get('enabled', True):
                pos = frame_config.get('position', [20, 70])
                color = tuple(frame_config.get('color', [255, 255, 255]))
                cv2.putText(res_img, f"Frame: {frame_count}", tuple(pos),
                           cv2.FONT_HERSHEY_SIMPLEX, frame_config.get('font_scale', 0.7),
                           color, frame_config.get('thickness', 2))

            # Skip Pose显示
            if not run_pose:
                skip_config = ui_config.get('skip_pose', {})
                if skip_config.get('enabled', True):
                    pos = skip_config.get('position', [200, 40])
                    color = tuple(skip_config.get('color', [200, 200, 200]))
                    cv2.putText(res_img, "(Skip Pose)", tuple(pos),
                               cv2.FONT_HERSHEY_SIMPLEX, skip_config.get('font_scale', 0.7),
                               color, skip_config.get('thickness', 2))

            # 缓冲区信息显示
            buffer_config = ui_config.get('buffer_info', {})
            if buffer_config.get('enabled', True):
                pos = buffer_config.get('position', [20, None])
                x_pos = pos[0]
                y_pos = height - 20 if pos[1] is None else pos[1]
                color = tuple(buffer_config.get('color', [200, 200, 200]))
                buffer_info = f"Buffers: {len(system.keypoint_buffers)}"
                for tid, buf in system.keypoint_buffers.items():
                    display_id = system.get_display_id(tid)
                    buffer_info += f" {display_id}:{len(buf['kpts'])}"
                cv2.putText(res_img, buffer_info, (x_pos, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, buffer_config.get('font_scale', 0.5),
                           color, buffer_config.get('thickness', 1))

        # 传出跌倒变量：每帧调用 fall_callback（若提供）
        if fall_callback is not None:
            try:
                fall_callback(system.get_fall_status())
            except Exception as e:
                pass  # 避免回调异常影响主循环

        cv2.imshow(window_name, res_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # 打印调试信息
            print(f"\n[调试信息] 帧数: {frame_count}")
            print(f"[调试信息] 缓冲区数量: {len(system.keypoint_buffers)}")
            for tid, buf in system.keypoint_buffers.items():
                print(f"[调试信息] Track ID {tid}: {len(buf['kpts'])}帧")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[完成] 推理结束")
    return system


def load_config(config_path='config.yaml'):
    """
    加载配置文件，提供完整的默认配置
    """
    # 默认配置（包含所有可配置项）
    default_config = {
        'models': {
            'vitpose': r'F:\Codes\实习\如身Robot\code\dataset\MyFD\easy_ViTPose\checkpoints\vitpose-s-coco.engine',
            'yolo': r'F:\Codes\实习\如身Robot\code\dataset\MyFD\easy_ViTPose\checkpoints\yolov8s.engine',
            'stgcn': 'stgcn.onnx'
        },
        'input': {
            'source': '0',
            'mode': 'auto',
            'rotate': 0
        },
        'output': {
            'image_path': None,
            'show_result': False,
            'window_name': 'Fall Detection'
        },
        'id_mapping': {
            'enabled': False,
            'auto_generate': True,
            'mapping': {}
        },
        'class_mapping': {
            '0': 'Fall',
            '1': 'Normal'
        },
        'detection': {
            'box_conf_threshold': 0.5,
            'pose_conf_threshold': 0.4,
            'dataset': 'coco',
            'min_box_size': 10,
            'max_persons': 1
        },
        'processing': {
            'pose_interval': 1,
            'window_size': 50,
            'prediction_interval': 10,
            'buffer_multiplier': 2,
            'min_frames_for_prediction': 30,
            'use_sliding_window': True,
            'fall_confirm_frames': 3  # 二次确认：连续多少帧预测为跌倒才判定为跌倒（>=1）
        },
        'device': {
            'use_cuda': True,
            'stgcn_provider': 'CPUExecutionProvider'
        },
        'drawing': {
            'bbox': {
                'enabled': True,
                'color': [255, 255, 0],
                'thickness': 2,
                'show_score': True,
                'show_stgcn_result': True,
                'stgcn_result_offset': [5, 25]
            },
            'keypoint': {
                'enabled': True,
                'color': [0, 255, 0],
                'radius': 4,
                'show_conf': True,
                'show_name': False,
                'conf_threshold': 0.4,
                'name_offset': [5, -10],
                'conf_offset': [0, -5]
            },
            'skeleton': {
                'enabled': True,
                'color': [255, 200, 0],
                'thickness': 2
            },
            'text': {
                'font': 'FONT_HERSHEY_SIMPLEX',
                'scale': 0.5,
                'thickness': 2,
                'color': [255, 255, 0]
            }
        },
        'display': {
            'result': {
                'enabled': True,
                'position': [20, 100],
                'line_spacing': 30,
                'font_scale': 0.7,
                'thickness': 2,
                'fall_color': [0, 0, 255],
                'normal_color': [0, 255, 0],
                'show_prob': True
            },
            'ui': {
                'enabled': True,
                'fps': {
                    'enabled': True,
                    'position': [20, 40],
                    'font_scale': 1.0,
                    'thickness': 2,
                    'color': [0, 255, 0]
                },
                'frame_count': {
                    'enabled': True,
                    'position': [20, 70],
                    'font_scale': 0.7,
                    'thickness': 2,
                    'color': [255, 255, 255]
                },
                'skip_pose': {
                    'enabled': True,
                    'position': [200, 40],
                    'font_scale': 0.7,
                    'thickness': 2,
                    'color': [200, 200, 200]
                },
                'buffer_info': {
                    'enabled': True,
                    'position': [20, None],
                    'font_scale': 0.5,
                    'thickness': 1,
                    'color': [200, 200, 200]
                }
            }
        }
    }
    
    if not os.path.exists(config_path):
        print(f"[警告] 配置文件 {config_path} 不存在，使用默认配置")
        return default_config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        file_config = yaml.safe_load(f)
    
    # 深度合并配置（文件配置覆盖默认配置）
    def deep_merge(default, override):
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = deep_merge(default_config, file_config)
    return config


def determine_mode(config):
    """
    根据配置确定运行模式
    """
    input_source = config['input']['source']
    mode = config['input']['mode']
    
    if mode == 'image':
        return 'image'
    elif mode == 'video':
        return 'video'
    elif mode == 'camera':
        return 'video'
    elif mode == 'auto':
        # 自动识别
        if input_source.isdigit():
            return 'video'  # 摄像头
        elif os.path.isfile(input_source):
            ext = os.path.splitext(input_source)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                return 'image'
            else:
                return 'video'  # 视频文件
        else:
            return 'video'  # 默认视频模式
    else:
        return 'video'  # 默认视频模式


if __name__ == "__main__":
    # 支持命令行参数（可选，优先级高于配置文件）
    parser = argparse.ArgumentParser(description='跌倒检测推理系统')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径（默认: config.yaml）')
    parser.add_argument('--input', type=str, default=None,
                       help='覆盖配置文件中的输入源')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 如果命令行指定了input，覆盖配置
    if args.input:
        config['input']['source'] = args.input

    try:
        import torch_tensorrt
    except ImportError:
        print("[警告] torch_tensorrt未安装，Engine加载可能失败")

    # 检查STGCN路径
    if not config['models']['stgcn']:
        print("[错误] 配置文件中必须指定STGCN模型路径")
        exit(1)

    # 确定运行模式
    mode = determine_mode(config)
    
    print(f"[配置] 使用配置文件: {args.config}")
    print(f"[配置] 运行模式: {mode}")
    print(f"[配置] 输入源: {config['input']['source']}")
    
    if mode == 'image':
        predict_image(config)
    else:
        def get_twice_ensure_fall(status):
            print(f"[二次跌倒确认] 下限={status['fall_consecutive']}  是否跌倒={status['fall']}")
        run_main(config, fall_callback=get_twice_ensure_fall)
