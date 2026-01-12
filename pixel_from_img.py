import cv2
import numpy as np
import os
import json  # 新增：用於處理 JSON

# === 設定顏色範圍字典 ===
HSV_RANGES = {
    'white':  [((0, 0, 220),    (180, 25, 255))],
    'blue':   [((100, 200, 100), (124, 255, 255))],
    'green':  [((35, 200, 100),  (85, 255, 255))],
    'yellow': [((20, 200, 100),  (34, 255, 255))],
    'black':  [((0, 0, 0),       (180, 255, 50))],
    'red':    [((0, 200, 100),   (10, 255, 255)), 
               ((156, 200, 100), (180, 255, 255))]
}

def extract_color_centers(video_path, frame_idx, target_color_name, min_area=10):
    """
    提取特定顏色的中心點座標，並回傳排序後的結果與視覺化圖像。
    """
    if target_color_name not in HSV_RANGES or not os.path.exists(video_path):
        print(f"錯誤：路徑不存在或顏色不支援。")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None, None, None

    # HSV 轉換與遮罩處理
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_frame.shape[:2], dtype="uint8")
    ranges = HSV_RANGES[target_color_name]
    for (lower, upper) in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8")))

    # 膨脹與輪廓查找
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    debug_img = frame.copy() 

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area: continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY)) # 先存入列表，稍後統一繪圖

    # === 關鍵排序 ===
    # 根據 X 座標由小到大排序 (由左至右)
    centers.sort(key=lambda p: p[0])

    # 依照排序後的順序進行繪圖 (確保圖像上的編號 1 對應到 JSON 的第一個點)
    for i, (cX, cY) in enumerate(centers):
        cv2.circle(debug_img, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(debug_img, str(i+1), (cX - 15, cY - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return centers, frame, debug_img

def save_points_to_json(centers, output_path):
    """
    將座標列表儲存為 JSON 檔案
    """
    data = {
        "total_count": len(centers),
        "points": []
    }

    # 依序填入資料
    for idx, (x, y) in enumerate(centers):
        data["points"].append({
            "id": idx + 1,
            "x": x,
            "y": y
        })

    try:
        # ensure_ascii=False 確保中文路徑或說明能正常顯示
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"成功儲存 JSON 至: {output_path}")
    except Exception as e:
        print(f"JSON 儲存失敗: {e}")

# === 主程式 ===
if __name__ == "__main__":
    video_file = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20251125棋盤格\New Patient Classification\New Patient\New Session\court_2.2129991.overlay.mp4"
    output_json_file = "output_points.json" # 輸出檔案名稱
    
    target_frame = 168
    target_color = 'red'
    
    print(f"正在處理...")
    centers, original, result_img = extract_color_centers(video_file, target_frame, target_color)

    if centers is not None and len(centers) > 0:
        print(f"共找到 {len(centers)} 個中心點。")
        
        # 1. 輸出 JSON
        save_points_to_json(centers, output_json_file)
        
        # 2. 顯示結果
        cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Result", 1024, 768)
        cv2.imshow("Final Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未找到目標顏色或執行失敗。")