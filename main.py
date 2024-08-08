import numpy as np
from scipy.optimize import curve_fit
from detectobject import DetectObject
import os
from physicalengine import PhysicsEngine
from PIL import Image

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
    

def fit_quadratic_function(t_values, y_values):
    """
    对给定的 t 值和 y 值拟合二次函数 y = at^2 + bt + c
    
    参数:
    t_values (list or array): t 的值列表
    y_values (list or array): y 的值列表
    
    返回:
    tuple: (a, b, c, r_squared)
    """
    def quadratic_function(t, a, b, c):
        return a * t**2 + b * t + c
    
    # 将输入转换为 numpy 数组
    t = np.array(t_values)
    y = np.array(y_values)
    
    # 使用 curve_fit 找到最佳拟合参数
    popt, _ = curve_fit(quadratic_function, t, y)
    # 解包参数
    a, b, c = popt
    
    # 计算 R-squared 值
    """
    对于变化大的数据，R-squared 值往往会更高，因为模型能解释更多的数据变异性。
    相比之下，变化小的数据可能会得到较低的 R-squared 值，即使实际拟合可能很好。
    """
    residuals = y - quadratic_function(t, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    fitted_function = lambda x: a * x**2 + b * x + c
    derivative_function = lambda x: 2 * a * x + b
    
    return a, b, c, r_squared, fitted_function, derivative_function


folder_path = '/Users/dong/Desktop/video/dataset/1/1/input/'
image_paths = get_image_paths(folder_path)

detector = DetectObject()
positions_of_circles = detector.detect(image_paths, ['circle'])

radius_list = []
for j in range(len(positions_of_circles['circle'][0])):
    sum_of_radius = 0
    for i in range(len(positions_of_circles['circle'])):
            sum_of_radius = sum_of_radius + positions_of_circles['circle'][i][j][2]
    radius_list.append(sum_of_radius/len(positions_of_circles['circle']))
desired_output, time_sequence = detector.row_data(positions_of_circles['circle'])

def get_image_dimensions(image_path):
    """
    This function takes the path to an image file as input and returns the dimensions (width, height) of the image.
    
    :param image_path: str, path to the image file
    :return: tuple, (width, height) of the image
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
image_path = '/Users/dong/Desktop/video/dataset/1/1/input/frame_00000.jpg'
dimensions = get_image_dimensions(image_path)

simulator = PhysicsEngine(width = dimensions[0], height = dimensions[1], gravity = (0,0))


for i in range(len(desired_output)):
    x_velocity = 0
    x_r_squared = 0
    x_acceleration = 0
    for j in range(len(desired_output[i])):
        list1 = time_sequence[i][j]
        list2 = desired_output[i][j]
        a, b, c, r_squared, fitted_func, derivate_func = fit_quadratic_function(list1, list2)
        #print(desired_output[i][0][0])
        #print(desired_output[i][1][0])
        if j == 0:
            x_velocity = b
            x_r_squared = r_squared
            x_acceleration = a * 2
        else:
            if r_squared > 0.5 and x_r_squared > 0.5:
                simulator.add_ball(position = (desired_output[i][0][0],desired_output[i][1][0]), radius = radius_list[i], mass = 20, velocity = (x_velocity, b), acceleration=(x_acceleration, a * 2))
            else:
                simulator.add_ball(position = (desired_output[i][0][0],desired_output[i][1][0]), radius = radius_list[i], mass = 20, velocity = (0, 0), acceleration=(0,0))
        #print(desired_output[i][0][0])
        #print(f"最佳拟合参数：")
        #print(f"a = {a:.8f}")
        #print(f"b = {b:.8f}")
        #print(f"c = {c:.8f}")
        #print(f"R-squared: {r_squared:.8f}")
        #print("\使用拟合函数计算值：")
        for k in range(len(list1)):
            y_fitted = fitted_func(list1[k])
            #print(f"t = {list1[k]}, y_fitted = {y_fitted:.8f}, y_original = {list2[k]}")

simulator.run()
