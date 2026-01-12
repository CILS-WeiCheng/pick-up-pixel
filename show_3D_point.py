import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_3d_points(json_file):
    """讀取 JSON 文件中的 3D 點座標"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    labels = []
    for key in sorted(data.keys()):
        points.append(data[key])
        labels.append(key)
    
    return np.array(points), labels

def plot_3d_points(points, labels):
    """繪製 3D 點並支持旋轉視角，並標示出原點位置"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製所有點
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='blue', s=100, alpha=0.7, edgecolors='black', linewidths=1)
    
    # 標註每個點
    for i, label in enumerate(labels):
        ax.text(points[i, 0], points[i, 1], points[i, 2], 
                f'  {label}', fontsize=9)
    
    # 顯示原點
    ax.scatter([0], [0], [0], c='red', s=120, marker='o', label='origin')
    ax.text(0, 0, 0, 'Left Camera', color='red', fontsize=11, fontweight='bold')

    # 設置座標軸標籤
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # 設置標題
    ax.set_title('3D Points', fontsize=14, pad=20)
    
    # 設置網格
    ax.grid(True, alpha=0.3)
    
    # 設置相等的縱橫比（可選，讓圖形更真實）
    # 計算範圍以設置相等的比例
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 繪製 z = 0 平面
    x_range = np.linspace(mid_x - max_range, mid_x + max_range, 20)
    y_range = np.linspace(mid_y - max_range, mid_y + max_range, 20)
    X_plane, Y_plane = np.meshgrid(x_range, y_range)
    Z_plane = np.zeros_like(X_plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='gray', 
                    label='z = 0 plane')
    
    # 繪製座標軸方向箭頭 (+x, +y, +z)
    arrow_length = max_range * 0.3  # 箭頭長度為範圍的 30%
    
    # +X 方向箭頭（紅色）
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', 
              arrow_length_ratio=0.15, linewidth=2, label='+X')
    ax.text(arrow_length * 1.1, 0, 0, '+X', color='red', fontsize=11, fontweight='bold')
    
    # +Y 方向箭頭（綠色）
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', 
              arrow_length_ratio=0.15, linewidth=2, label='+Y')
    ax.text(0, arrow_length * 1.1, 0, '+Y', color='green', fontsize=11, fontweight='bold')
    
    # +Z 方向箭頭（藍色）
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', 
              arrow_length_ratio=0.15, linewidth=2, label='+Z')
    ax.text(0, 0, arrow_length * 1.1, '+Z', color='blue', fontsize=11, fontweight='bold')
    
    # 顯示圖例，方便辨識原點
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 讀取 3D 點數據
    json_file = './origin_img/final_transformed_points_2.json'
    points, labels = load_3d_points(json_file)
    
    print(f"成功載入 {len(points)} 個 3D 點")
    print(f"X 範圍: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y 範圍: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z 範圍: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # 繪製 3D 點
    plot_3d_points(points, labels)

