import cv2
import numpy as np
import json

# ==========================================
# 1. 設定路徑與參數
# ==========================================
# 請確認這裡是原始圖片
LEFT_IMG_PATH = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_校正後圖片\court\origin_L.jpg"
RIGHT_IMG_PATH = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_校正後圖片\court\origin_R.jpg"
SAVE_JSON_PATH = "correspond_points_epipolar.json"

# 標定參數路徑
STEREO_PARAM_PATH = './20251125_exp1_雙目參數檔/exp1/stereo_rt_result_1_2.npz'
LEFT_PARAM_PATH = './20251125_exp1_雙目參數檔/exp1/20251125_left_single_1_2.npz'
RIGHT_PARAM_PATH = './20251125_exp1_雙目參數檔/exp1/20251125_right_single_1_2.npz'

# 放大鏡設定
ZOOM_SCALE = 5        # 放大倍率
ZOOM_WIN_SIZE = 300   # 放大鏡視窗大小 (像素)

def imread_unicode(path):
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ==========================================
# 2. 載入參數與計算 Essential Matrix
# ==========================================
try:
    left_data = np.load(LEFT_PARAM_PATH)
    right_data = np.load(RIGHT_PARAM_PATH)
    stereo_data = np.load(STEREO_PARAM_PATH)

    mtxL, distL = left_data['camera_matrix'], left_data['dist_coeffs']
    mtxR, distR = right_data['camera_matrix'], right_data['dist_coeffs']
    R, T = stereo_data['R'], stereo_data['T']
    
    # 計算本質矩陣 E
    T_flat = T.flatten()
    Tx = np.array([
        [0, -T_flat[2], T_flat[1]],
        [T_flat[2], 0, -T_flat[0]],
        [-T_flat[1], T_flat[0], 0]
    ])
    E = Tx @ R
    print("本質矩陣 E 計算完成。")

except Exception as e:
    print(f"載入參數失敗: {e}")
    exit()

# ==========================================
# 3. 核心：畫極線曲線 & 放大鏡
# ==========================================
def draw_epipolar_curve(img_target, pt_source, mtx_src, dist_src, mtx_tgt, dist_tgt, E_mat, is_source_left):
    """在目標圖上畫出對應的極線曲線 (考慮畸變)"""
    # 1. 將來源點去畸變，轉為歸一化座標
    pt_src_arr = np.array([[[pt_source[0], pt_source[1]]]], dtype=np.float32)
    pt_src_norm = cv2.undistortPoints(pt_src_arr, mtx_src, dist_src, P=None)
    x_src, y_src = pt_src_norm[0, 0]
    v_src = np.array([x_src, y_src, 1.0])

    # 2. 計算目標圖上的極線方程式
    if is_source_left:
        line_params = E @ v_src # l_R = E * x_L
    else:
        line_params = E.T @ v_src # l_L = E^T * x_R
        
    a, b, c = line_params

    # 3. 採樣極線
    points_norm_line = []
    if abs(b) > abs(a): 
        xs = np.linspace(-3.0, 3.0, 200)
        for x in xs:
            y = -(a * x + c) / b
            points_norm_line.append([x, y, 1.0])
    else:
        ys = np.linspace(-3.0, 3.0, 200)
        for y in ys:
            x = -(b * y + c) / a
            points_norm_line.append([x, y, 1.0])

    if len(points_norm_line) == 0: return

    points_3d = np.array(points_norm_line, dtype=np.float32)

    # 4. 投影回目標圖片 (自動加畸變)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx_tgt, dist_tgt)
    img_points = img_points.reshape(-1, 2).astype(np.int32)
    
    # 5. 過濾並畫線
    h, w = img_target.shape[:2]
    valid_points = []
    for p in img_points:
        if -100 < p[0] < w+100 and -100 < p[1] < h+100:
            valid_points.append(p)
            
    if len(valid_points) > 1:
        cv2.polylines(img_target, [np.array(valid_points)], False, (0, 255, 0), 2)

def update_zoom_window(img, x, y):
    """更新放大鏡視窗內容"""
    h, w = img.shape[:2]
    
    # 計算在原圖上要裁切的範圍 (ROI)
    # ROI大小 = 放大視窗大小 / 放大倍率
    roi_w = int(ZOOM_WIN_SIZE / ZOOM_SCALE)
    roi_h = int(ZOOM_WIN_SIZE / ZOOM_SCALE)
    
    # 確保不會切出邊界
    x1 = max(0, x - roi_w // 2)
    y1 = max(0, y - roi_h // 2)
    x2 = min(w, x + roi_w // 2)
    y2 = min(h, y + roi_h // 2)
    
    # 裁切影像
    roi = img[y1:y2, x1:x2]
    
    # 如果靠邊導致裁切區域不足，補黑邊保持視窗大小一致
    if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
        temp = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        # 計算貼上的起始點
        tx = 0 if x - roi_w // 2 >= 0 else (roi_w // 2 - x)
        ty = 0 if y - roi_h // 2 >= 0 else (roi_h // 2 - y)
        th, tw = roi.shape[:2]
        temp[ty:ty+th, tx:tx+tw] = roi
        roi = temp
        
    # 放大影像 (使用最近鄰插值以保持像素顆粒感，方便對準)
    zoom_img = cv2.resize(roi, (ZOOM_WIN_SIZE, ZOOM_WIN_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # 畫中心十字準星
    center = ZOOM_WIN_SIZE // 2
    cv2.line(zoom_img, (center, 0), (center, ZOOM_WIN_SIZE), (0, 0, 255), 1)
    cv2.line(zoom_img, (0, center), (ZOOM_WIN_SIZE, center), (0, 0, 255), 1)
    
    cv2.imshow("Zoom View", zoom_img)

# ==========================================
# 4. 互動變數與回呼函式
# ==========================================
pt_L = None
pt_R = None
points_list = []

imgL_origin = imread_unicode(LEFT_IMG_PATH)
imgR_origin = imread_unicode(RIGHT_IMG_PATH)

imgL_display = imgL_origin.copy()
imgR_display = imgR_origin.copy()
imgL_clean = imgL_display.copy()
imgR_clean = imgR_display.copy()

is_swapped = False

def mouse_callback_L(event, x, y, flags, param):
    global pt_L, imgL_display, imgR_display
    
    # 處理放大鏡
    if event == cv2.EVENT_MOUSEMOVE:
        update_zoom_window(imgL_display, x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pt_L = (x, y)
        print(f"[左視窗] 選取點: {pt_L}")
        
        # 重置並畫點
        imgL_display = imgL_clean.copy()
        imgR_display = imgR_clean.copy()
        cv2.circle(imgL_display, pt_L, 5, (0, 0, 255), -1)
        
        # 畫對應的極線曲線
        if not is_swapped:
            draw_epipolar_curve(imgR_display, pt_L, mtxL, distL, mtxR, distR, E, is_source_left=True)
        else:
            draw_epipolar_curve(imgR_display, pt_L, mtxR, distR, mtxL, distL, E, is_source_left=False)

        cv2.imshow("Left Window", imgL_display)
        cv2.imshow("Right Window", imgR_display)

def mouse_callback_R(event, x, y, flags, param):
    global pt_R, points_list, imgR_display, imgL_clean, imgR_clean
    
    # 處理放大鏡
    if event == cv2.EVENT_MOUSEMOVE:
        update_zoom_window(imgR_display, x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if pt_L is None:
            print("請先在左視窗選點！")
            return
            
        pt_R = (x, y)
        print(f"[右視窗] 選取點: {pt_R} -> 配對暫存")
        
        cv2.circle(imgR_display, pt_R, 5, (0, 0, 255), -1)
        cv2.imshow("Right Window", imgR_display)
        
        # 儲存
        if not is_swapped:
            final_pL, final_pR = pt_L, pt_R
        else:
            final_pL, final_pR = pt_R, pt_L
            
        points_list.append((final_pL, final_pR))
        
        # 烙印
        cv2.circle(imgL_clean, pt_L, 5, (0, 255, 0), -1)
        cv2.circle(imgR_clean, pt_R, 5, (0, 255, 0), -1)
        idx = len(points_list)
        cv2.putText(imgL_clean, str(idx), pt_L, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(imgR_clean, str(idx), pt_R, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 刷新移除極線
        imgL_display[:] = imgL_clean[:]
        imgR_display[:] = imgR_clean[:]
        cv2.imshow("Left Window", imgL_display)
        cv2.imshow("Right Window", imgR_display)

def swap_images():
    global imgL_clean, imgR_clean, imgL_display, imgR_display, is_swapped, points_list
    print("交換左右圖片顯示...")
    is_swapped = not is_swapped
    points_list = [] # 清空
    
    if not is_swapped:
        imgL_clean = imgL_origin.copy()
        imgR_clean = imgR_origin.copy()
    else:
        imgL_clean = imgR_origin.copy()
        imgR_clean = imgL_origin.copy()
        
    imgL_display = imgL_clean.copy()
    imgR_display = imgR_clean.copy()
    
    cv2.imshow("Left Window", imgL_display)
    cv2.imshow("Right Window", imgR_display)

# ==========================================
# 5. 主迴圈
# ==========================================
print("=== 極線曲線輔助工具 v4 (含放大鏡) ===")
print("  - 移動滑鼠: 自動顯示放大鏡 (Zoom View)")
print("  - 左鍵: 選點")
print("  - 'x': 交換左右圖")
print("  - 's': 儲存")
print("  - 'q': 離開")

# 建立視窗
cv2.namedWindow("Left Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Left Window", 960, 540)
cv2.setMouseCallback("Left Window", mouse_callback_L)

cv2.namedWindow("Right Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Right Window", 960, 540)
cv2.setMouseCallback("Right Window", mouse_callback_R)

# 建立放大鏡視窗
cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Zoom View", ZOOM_WIN_SIZE, ZOOM_WIN_SIZE)

cv2.imshow("Left Window", imgL_display)
cv2.imshow("Right Window", imgR_display)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(points_list) > 0:
            output_data = {
                f"point{i+1}": {"left": list(p[0]), "right": list(p[1])}
                for i, p in enumerate(points_list)
            }
            with open(SAVE_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            print(f"已儲存 {len(points_list)} 組至 {SAVE_JSON_PATH}")
        break
    elif key == ord('q'):
        break
    elif key == ord('x'):
        swap_images()

cv2.destroyAllWindows()