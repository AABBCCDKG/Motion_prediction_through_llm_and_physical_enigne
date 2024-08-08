import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from scipy.optimize import curve_fit
from detectobject import DetectObject
from PIL import Image
import cv2

def get_image_paths(folder_path):
        if not os.path.exists(folder_path):
            raise ValueError(f"The folder path {folder_path} does not exist.")
        
        # 获取文件夹中的所有文件名
        all_files = os.listdir(folder_path)
        
        # 过滤出符合条件的文件名
        image_files = [f for f in all_files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        # 按文件名排序
        image_files.sort()

        # 构建完整路径
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        
        return image_paths


def video_to_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps} 帧总数: {frame_count}")
    
    # 初始化帧计数器
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存帧图像
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_idx += 1
    
    # 释放视频捕捉对象
    cap.release()
    print(f"视频帧提取完成，共提取 {frame_idx} 帧")

def draw_frame(circles, frame_number, save_path):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=50)
    ax = fig.add_subplot(111)
    
    # 设置坐标系
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)  # 注意这里y轴的顺序是相反的
    ax.set_position([0, 0, 1, 1])
    # 移除坐标轴
    ax.axis('off')
   
    for x, y, r in circles:
        circle = Circle((x, y), r, fill=False, ec='black')
        ax.add_artist(circle)
    
    # 使用自定义的方式保存图片，以确保大小正确
    canvas = FigureCanvasAgg(fig)
    
    canvas.draw()
    image = canvas.buffer_rgba()
    
    from PIL import Image
    image = Image.frombytes('RGBA', canvas.get_width_height(), image)
    image = image.convert('RGB')
    width, height = canvas.get_width_height()
    
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存图片到指定路径
    image.save(os.path.join(save_path, f'frame_{frame_number:05d}.jpg'))
    
    plt.close(fig)

def draw_all_frames(frames, save_path):
    for m, frame in enumerate(frames):
        draw_frame(frame, m, save_path)
        m += 1


#For training dataset

for i in range(1,21):
    video_path = f'/Users/dong/Desktop/Max/video/data/noclassification/{i}.mp4'
    output_folder = f'/Users/dong/Desktop/max/video/mapping/dataset/training/A/{i}/'
    video_to_frames(video_path, output_folder)
    folder_path = f'/Users/dong/Desktop/max/video/mapping/dataset/training/A/{i}/'
    image_paths = get_image_paths(folder_path)

    detector = DetectObject()
    positions_of_circles = detector.detect(image_paths, ['circle'])

    radius_list = []
    for j in range(len(positions_of_circles['circle'][0])):
        sum_of_radius = 0
        for m in range(len(positions_of_circles['circle'])):
                sum_of_radius = sum_of_radius + positions_of_circles['circle'][m][j][2]
        radius_list.append(sum_of_radius/len(positions_of_circles['circle']))

    positions_list = positions_of_circles['circle']
    save_path = f'/Users/dong/Desktop/Max/video/mapping/dataset/training/B/{i}/'
    draw_all_frames(positions_list, save_path)

