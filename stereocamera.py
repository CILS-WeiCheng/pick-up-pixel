import cv2
import numpy as np
import glob
import os

# --- 1. 核心設定：定義棋盤格和物理尺寸 ---
# 這是最關鍵的步驟，請務必精確測量！

# 棋盤格內角點的數量 (寬 x 高)
CHESSBOARD_SIZE = (9, 6)
# 棋盤格每一格的「實際邊長」，單位必須是「公尺」
SQUARE_SIZE_METERS = 0.1

# 準備 3D 理想座標 (objp)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS

# 迭代的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

def calibrate_single_camera(image_dir, camera_name="相機", batch_size=15):
    """
    標定單個攝影機的內參和畸變係數。
    
    參數:
        image_dir: 包含標定影像的資料夾路徑
        camera_name: 相機名稱（用於顯示訊息）
        batch_size: 批次大小，用於分批處理標定（預設為 15）
    
    返回:
        (mtx, dist, image_size, ret) 或 None
    """
    print(f"\n開始標定 {camera_name}...")
    
    # 讀取所有標定影像
    images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    
    if len(images) == 0:
        print(f"錯誤：在 {image_dir} 中找不到影像。")
        return None
    
    # 用來儲存所有影像的 3D 點和 2D 點
    objpoints = []  # 3D 點 (世界座標)
    imgpoints = []  # 2D 點 (影像座標)
    
    image_size = None
    
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"警告：無法讀取影像 {img_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]  # (寬, 高)
        
        # 尋找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            objpoints.append(objp)
            
            # 提高 2D 角點的精度
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_subpix)
            
            print(f"處理中：第 {i+1}/{len(images)} 張影像... 成功找到角點。")
        else:
            print(f"處理中：第 {i+1}/{len(images)} 張影像... 失敗，跳過。")
    
    if not objpoints:
        print(f"錯誤：在 {camera_name} 的所有影像中都無法找到完整的棋盤格。")
        return None
    
    print(f"\n{camera_name} 標定完成，共使用 {len(objpoints)} 張有效的影像。")
    print("--------------------------------------------------")
    
    # 設定旗標以使用 8 參數的 Rational Model，更適合廣角鏡頭
    calib_flags = cv2.CALIB_RATIONAL_MODEL
    
    # 分批處理標定
    print(f"正在分批計算 {camera_name} 內參（批次大小：{batch_size}）...")
    all_rvecs = []
    all_tvecs = []
    mtx = None
    dist = None
    ret = None
    
    for i in range(0, len(objpoints), batch_size):
        batch_end = min(i + batch_size, len(objpoints))
        batch_objpoints = objpoints[i:batch_end]
        batch_imgpoints = imgpoints[i:batch_end]
        
        try:
            batch_ret, batch_mtx, batch_dist, batch_rvecs, batch_tvecs = cv2.calibrateCamera(
                batch_objpoints, batch_imgpoints, image_size, None, None,
                flags=calib_flags
            )
            
            # 收集所有批次的 rvecs 和 tvecs
            all_rvecs.extend(batch_rvecs)
            all_tvecs.extend(batch_tvecs)
            
            # 使用最後一批的結果作為最終結果（或者可以選擇誤差最小的批次）
            # 這裡使用最後一批的結果
            mtx = batch_mtx
            dist = batch_dist
            ret = batch_ret
            
            print(f"批次 {i//batch_size + 1}：處理了 {len(batch_objpoints)} 張影像，重投影誤差：{batch_ret:.6f} 像素")
            
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 標定計算錯誤：{e}")
            continue
    
    if mtx is None or dist is None:
        print(f"錯誤：{camera_name} 標定失敗。")
        return None
    
    # 最終使用所有資料進行一次完整標定（推薦方式）
    print(f"\n使用所有 {len(objpoints)} 張影像進行最終標定...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=calib_flags
    )
    
    print(f"{camera_name} 最終平均重投影誤差 (ret): {ret:.6f} 像素")
    print(f"{camera_name} 內參矩陣 (mtx):\n", mtx)
    print(f"{camera_name} 畸變係數 (dist):\n", dist)
    print("--------------------------------------------------")
    
    return mtx, dist, image_size, ret


def calibrate_stereo_rig(left_image_dir, right_image_dir, mtxL, distL, mtxR, distR, image_size):
    """
    雙目標定流程：計算左右相機之間的相對位置關係 (R, T)。
    此函式使用已經標定好的內參和畸變係數，專注於求解雙目外參。
    
    參數:
        left_image_dir: 左相機標定影像資料夾
        right_image_dir: 右相機標定影像資料夾
        mtxL, distL: 左相機內參和畸變係數（已標定）
        mtxR, distR: 右相機內參和畸變係數（已標定）
        image_size: 影像尺寸 (寬, 高)
    
    返回:
        (R, T, ret_Stereo) 或 None
    """
    print("\n開始雙目標定流程...")
    
    # 讀取所有成對的標定影像
    left_images = sorted(glob.glob(os.path.join(left_image_dir, '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(right_image_dir, '*.jpg')))
    
    if len(left_images) == 0 or len(right_images) == 0:
        print(f"錯誤：在 {left_image_dir} 或 {right_image_dir} 中找不到影像。")
        return None
    
    if len(left_images) != len(right_images):
        print("警告：左右相機的影像數量不一致，將只使用較少者。")
    
    # 用來儲存所有影像的 3D 點和 2D 點
    objpoints = []  # 3D 點 (世界座標)
    imgpoints_L = []  # 2D 點 (左相機影像)
    imgpoints_R = []  # 2D 點 (右相機影像)
    
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        img_L = cv2.imread(left_path)
        img_R = cv2.imread(right_path)
        
        if img_L is None or img_R is None:
            print(f"警告：無法讀取影像 {left_path} 或 {right_path}")
            continue
        
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        
        # 尋找棋盤格角點
        ret_L, corners_L = cv2.findChessboardCorners(gray_L, CHESSBOARD_SIZE, None)
        ret_R, corners_R = cv2.findChessboardCorners(gray_R, CHESSBOARD_SIZE, None)
        
        # 關鍵：必須兩張影像「同時」都找到角點，這一對才算有效
        if ret_L and ret_R:
            objpoints.append(objp)
            
            # 提高 2D 角點的精度
            corners_L_subpix = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
            imgpoints_L.append(corners_L_subpix)
            
            corners_R_subpix = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)
            imgpoints_R.append(corners_R_subpix)
            
            # --- 視覺化除錯：檢查角點順序 ---
            # 只在第一張有效的影像上執行
            if len(objpoints) == 1:
                print("\n*** 正在進行視覺檢查 ***")
                print("請檢查左右兩張圖中，綠色線條的「起點」和「方向」是否一致？")
                
                # 繪製角點和順序
                img_L_corners = np.copy(img_L)  # 複製影像以免影響原始影像
                img_R_corners = np.copy(img_R)
                cv2.drawChessboardCorners(img_L_corners, CHESSBOARD_SIZE, corners_L_subpix, ret_L)
                cv2.drawChessboardCorners(img_R_corners, CHESSBOARD_SIZE, corners_R_subpix, ret_R)
                
                # 疊加顯示
                debug_image = np.hstack((img_L_corners, img_R_corners))
                
                # 調整視窗大小以便查看
                cv2.imshow('Corner Detection Order - Press any key to continue', cv2.resize(debug_image, None, fx=0.8, fy=0.8))
                
                print("... 按下任意鍵繼續標定 ...")
                cv2.waitKey(0)
                cv2.destroyWindow('Corner Detection Order - Press any key to continue')
            # --- 視覺化除錯結束 ---
            
            # --- 如果順序不一致，請取消註解並修改以下行 ---
            # 例如，如果右相機的角點順序是顛倒的：
            # imgpoints_R.append(np.flip(corners_R_subpix, axis=0))
            # -------------------------------------------------
            
            print(f"處理中：第 {i+1}/{len(left_images)} 對影像... 成功找到角點。")
        else:
            print(f"處理中：第 {i+1}/{len(left_images)} 對影像... 失敗，跳過此對。")
    
    if not objpoints:
        print("錯誤：在所有影像對中都無法找到完整的棋盤格。")
        return None
    
    print(f"\n雙目標定完成，共使用 {len(objpoints)} 對有效的影像。")
    print("--------------------------------------------------")
    
    # --- 雙目標定：計算 R 和 T ---
    print("正在計算雙目外參 (R, T)...")
    
    # 使用已標定的內參作為初始猜測，允許微調
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    
    (ret_Stereo, mtxL_new, distL_new, mtxR_new, distR_new, R, T, E, F) = cv2.stereoCalibrate(
        objpoints,
        imgpoints_L,
        imgpoints_R,
        mtxL, distL,
        mtxR, distR,
        image_size,
        criteria=criteria_stereo,
        flags=flags
    )
    
    print("E (本質矩陣):\n", E)
    print("F (基礎矩陣):\n", F)
    print("R (左右相機之間的旋轉矩陣):\n", R)
    print("T (左右相機之間的平移矩陣):\n", T)
    print(f"雙目相機 平均重投影誤差 (ret_Stereo): {ret_Stereo:.6f} 像素")
    print("\n雙目標定完成！")
    
    # 計算基線 (僅供驗證)
    baseline_m = np.linalg.norm(T)
    print(f"計算出的基線 (Baseline) 距離: {baseline_m:.4f} 公尺")
    
    # 返回更新後的內參（可能經過微調）和雙目外參
    return mtxL_new, distL_new, mtxR_new, distR_new, R, T, ret_Stereo

def get_rectification_maps(mtxL, distL, mtxR, distR, R, T, image_size):
    """
    步驟 3 和 4：計算立體校正並建立映射表
    """
    print("\n步驟 3：正在計算立體校正參數 (R1, R2, P1, P2, Q)...")

    # --- 步驟 3：立體校正 (stereoRectify) ---
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, image_size, R, T, 
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )
    # alpha = 0 時，不裁切影像，無效區域會顯示為黑色。
    # alpha = -1 時，裁切影像，只保留有效像素。
    # alpha = 1 時，保留所有原始像素，無效區域會顯示為黑色。
    
    print("R1 (左相機校正旋轉矩陣):\n", R1)
    print("R2 (右相機校正旋轉矩陣):\n", R2)
    print("P1 (左相機新投影矩陣):\n", P1)
    print("P2 (右相機新投影矩陣):\n", P2)
    print("Q (深度圖的映射矩陣):\n", Q)

    print("\n步驟 4：正在建立校正映射表 (initUndistortRectifyMap)...")
    # --- 步驟 4：建立映射表 (initUndistortRectifyMap) ---
    left_map_1, left_map_2 = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map_1, right_map_2 = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, image_size, cv2.CV_16SC2
    )
    
    print("映射表建立完成。")
    return left_map_1, left_map_2, right_map_1, right_map_2, roi1, roi2


def main():
    # --- 執行標定 ---
    # 單攝影機標定使用的圖片資料夾（用於計算內參和畸變係數）
    LEFT_SINGLE_PATH = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\left\selected_15"
    RIGHT_SINGLE_PATH = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\right\selected_15"
    
    # 雙目標定使用的圖片資料夾（用於計算 R 和 T）
    LEFT_CALIB_PATH = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\left\stereo_image_15"
    RIGHT_CALIB_PATH = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\right\stereo_image_15"

    # --- 步驟 1：單攝影機標定（內參和畸變係數）---
    print("=" * 60)
    print("步驟 1：單攝影機標定（內參和畸變係數）")
    print("=" * 60)
    
    try:
        # 標定左相機
        left_result = calibrate_single_camera(LEFT_SINGLE_PATH, "左相機", batch_size=15)
        if left_result is None:
            print("左相機標定失敗，無法繼續。")
            return
        mtxL, distL, image_size_L, ret_L = left_result
        
        # 標定右相機
        right_result = calibrate_single_camera(RIGHT_SINGLE_PATH, "右相機", batch_size=15)
        if right_result is None:
            print("右相機標定失敗，無法繼續。")
            return
        mtxR, distR, image_size_R, ret_R = right_result
        
        # 檢查兩個相機的影像尺寸是否一致
        if image_size_L != image_size_R:
            print(f"警告：左右相機影像尺寸不一致！左：{image_size_L}，右：{image_size_R}")
            print("將使用左相機的影像尺寸。")
        image_size = image_size_L
        
        # 檢查重投影誤差
        if ret_L > 1.0:
            print(f"\n*** 警告：左相機平均重投影誤差 {ret_L:.4f} 像素過高 (應 < 1.0) ***")
        if ret_R > 1.0:
            print(f"\n*** 警告：右相機平均重投影誤差 {ret_R:.4f} 像素過高 (應 < 1.0) ***")
            
    except cv2.error as e:
        print(f"OpenCV 單攝影機標定時發生錯誤：{e}")
        print("請檢查您的棋盤格設定 (CHESSBOARD_SIZE) 是否與影像中的角點數一致。")
        return
    
    # --- 步驟 2：雙目標定（R 和 T）---
    print("\n" + "=" * 60)
    print("步驟 2：雙目標定（計算 R 和 T）")
    print("=" * 60)
    
    try:
        stereo_results = calibrate_stereo_rig(
            LEFT_CALIB_PATH, RIGHT_CALIB_PATH,
            mtxL, distL, mtxR, distR, image_size
        )
        if stereo_results is None:
            print("雙目標定失敗，無法繼續。")
            return
    except cv2.error as e:
        print(f"OpenCV 雙目標定時發生錯誤：{e}")
        print("請檢查您的棋盤格設定 (CHESSBOARD_SIZE) 是否與影像中的角點數一致。")
        return
    
    (mtxL_final, distL_final, mtxR_final, distR_final, R, T, ret_Stereo) = stereo_results
    print(f"mtxL : {mtxL}\n")
    print(f"distL : {distL}\n")
    print(f"mtxR : {mtxR}\n")
    print(f"distR : {distR}\n")
    print(f"mtxL_final : {mtxL_final}\n")
    print(f"distL_final : {distL_final}\n")
    print(f"mtxR_final : {mtxR_final}\n")
    print(f"distR_final : {distR_final}\n")
    print(f"R : {R}\n")
    print(f"T : {T}\n")
    print(f"ret_Stereo : {ret_Stereo}\n")

    
    # 檢查雙目標定的重投影誤差
    if ret_Stereo > 1.0:
        print(f"\n*** 警告：雙目相機平均重投影誤差 {ret_Stereo:.4f} 像素過高 (應 < 1.0) ***")
    
    # --- 步驟 3 和 4：取得映射表 ---
    print("\n" + "=" * 60)
    print("步驟 3 和 4：計算立體校正映射表")
    print("=" * 60)
    
    try:
        maps = get_rectification_maps(mtxL_final, distL_final, mtxR_final, distR_final, R, T, image_size)
        (left_map_1, left_map_2, right_map_1, right_map_2, roi1, roi2) = maps
    except cv2.error as e:
        print(f"OpenCV 立體校正時發生錯誤：{e}")
        return

    # --- 步驟 5：應用映射表 (Remap) ---
    print("\n" + "=" * 60)
    print("步驟 5：應用映射表校正範例影像")
    print("=" * 60)
    
    # 替換為您想要校正的「原始」影像 (例如羽球場照片)
    LEFT_TEST_IMG = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\left\best_15_selected\best_image_01.jpg"
    RIGHT_TEST_IMG = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\right\best_15_selected\best_image_01.jpg"
    
    try:
        img_L_orig = cv2.imread(LEFT_TEST_IMG)
        img_R_orig = cv2.imread(RIGHT_TEST_IMG)
        if img_L_orig is None or img_R_orig is None:
            raise FileNotFoundError("無法讀取範例影像。")
            
        # 執行 Remap
        left_rectified = cv2.remap(img_L_orig, left_map_1, left_map_2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(img_R_orig, right_map_1, right_map_2, cv2.INTER_LINEAR)

        # (可選) 根據 ROI 裁切
        # x, y, w, h = roi1
        # left_rectified = left_rectified[y:y+h, x:x+w]
        # x, y, w, h = roi2
        # right_rectified = right_rectified[y:y+h, x:x+w]

        x1, y1, w1, h1 = roi1
        x2, y2, w2, h2 = roi2
        # 計算右視角的右上角
        x2 = x2+w2
        # 3. 計算「交集」的寬和高
        common_w = min(w1, w2)
        common_h = min(h1, h2)

        if common_w > 0 and common_h > 0:
            # 使用這個「共同的 ROI」來裁切兩張校正後的影像
            # (注意：這裡的 common_x, common_y 是相對於校正後影像的)
            left_rect_cropped = left_rectified[y1:y1+common_h, x1:x1+common_w]
            right_rect_cropped = right_rectified[y2:y2+common_h, (x2-common_w):x2]
        else:
            print("ROI 沒有交集！")

        print("校正完成！")

        # --- 顯示或儲存結果 ---
        
        # 為了方便比較，將兩張校正後的影像並排
        combined_image = np.hstack((left_rectified, right_rectified))
        
        # 畫上水平極線 (Epipolar Lines) 以供驗證
        num_lines = 20
        line_color = (0, 255, 0)
        for y in np.linspace(0, combined_image.shape[0], num_lines, dtype=int):
            if y == combined_image.shape[0]: y -= 1 # 避免剛好在邊界上
            cv2.line(combined_image, (0, y), (combined_image.shape[1], y), line_color, 1)

        output_filename = 'stereo_rectified_output.jpg'
        cv2.imwrite(output_filename, combined_image)
        print(f"已儲存並排的校正影像 (含水平線) 至：{output_filename}")

        # (可選) 如果您在 Jupyter 環境之外執行，可以用 cv2.imshow 顯示
        # cv2.imshow('Rectified Stereo Pair with Epipolar Lines', cv2.resize(combined_image, None, fx=0.4, fy=0.4))
        # print("\n按任意鍵關閉視窗...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"錯誤：找不到範例影像路徑。")
    except cv2.error as e:
        print(f"OpenCV 執行 remap 時發生錯誤：{e}")


if __name__ == "__main__":
    main()