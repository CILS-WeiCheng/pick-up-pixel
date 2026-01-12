import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as Rscipy

class StereoVisionSystem:
    def __init__(self, left_calib_path, right_calib_path, stereo_calib_path):
        """
        初始化雙目視覺系統
        """
        self.left_calib_path = left_calib_path
        self.right_calib_path = right_calib_path
        self.stereo_calib_path = stereo_calib_path
        
        # 參數容器
        self.mtxL, self.distL = None, None
        self.mtxR, self.distR = None, None
        self.R, self.T = None, None
        self.P_L, self.P_R = None, None
        
        # 資料容器
        self.img_points = {}       # 像素座標
        self.raw_3d_points = {}    # 三角測量後的原始 3D 座標 (左相機座標系)
        self.vicon_3d_points = {}  # VICON Ground Truth
        self.aligned_points = {}   # 經 SVD 轉換後對齊的 3D 座標

    @staticmethod
    def imread_unicode(file_path):
        """讀取包含中文路徑的圖片"""
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取圖片: {file_path}")
        return img

    def load_parameters(self):
        """載入單目與雙目參數並建構投影矩陣"""
        print("--- 載入校正參數 ---")
        # 1. 載入單目參數
        left_data = np.load(self.left_calib_path, allow_pickle=True)
        self.mtxL = left_data['camera_matrix']
        self.distL = left_data['dist_coeffs']
        
        right_data = np.load(self.right_calib_path, allow_pickle=True)
        self.mtxR = right_data['camera_matrix']
        self.distR = right_data['dist_coeffs']
        
        # 2. 載入雙目外參
        stereo_data = np.load(self.stereo_calib_path, allow_pickle=True)
        self.R = stereo_data['R']
        self.T = stereo_data['T']
        
        print(f"Baseline: {np.linalg.norm(self.T):.4f} m")

        # 3. 建構投影矩陣 P = K @ [R|T]
        # 左相機為世界座標原點 [I|0]
        self.P_L = self.mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # 右相機相對位置 [R|T]
        self.P_R = self.mtxR @ np.hstack((self.R, self.T))
        
        print("投影矩陣建構完成。")

    def get_image_points(self, mode='json', json_path=None, left_img_path=None, right_img_path=None, num_points=15, save_path="manual_points.json"):
        """
        獲取圖像座標點
        :param mode: 'json' (讀檔) 或 'manual' (手動選點)
        :param save_path: 手動選點後要儲存的 JSON 檔名 (預設為 manual_points.json)
        """
        if mode == 'json' and json_path:
            print(f"--- 從 JSON 載入點位: {json_path} ---")
            with open(json_path, 'r', encoding='utf-8') as f:
                self.img_points = json.load(f)
        
        elif mode == 'manual' and left_img_path and right_img_path:
            print(f"--- 啟動手動選點模式 (目標: {num_points} 點) ---")
            
            # 讀取圖片
            img_L = self.imread_unicode(left_img_path)
            img_R = self.imread_unicode(right_img_path)
            
            # 執行互動選點
            pts_L = self._interactive_select(img_L, "Select Left Image", num_points)
            pts_R = self._interactive_select(img_R, "Select Right Image", num_points)
            
            # 檢查點數是否一致
            if len(pts_L) != num_points or len(pts_R) != num_points:
                print(f"警告: 選取的點數不足 (左: {len(pts_L)}, 右: {len(pts_R)})，請重新執行。")
                return

            # 建構符合 POINTS_JSON 格式的字典
            # 格式: {"point1": {"left": [u, v], "right": [u, v]}, ...}
            self.img_points = {}
            for i in range(num_points):
                self.img_points[f"point{i+1}"] = {
                    "left": [int(pts_L[i][0]), int(pts_L[i][1])],   # 轉為 int 避免 JSON 報錯
                    "right": [int(pts_R[i][0]), int(pts_R[i][1])]
                }
            
            # 自動儲存為 JSON
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(self.img_points, f, indent=4, ensure_ascii=False)
                print(f"選點完成！座標已儲存至: {save_path}")
                print("格式已對齊，下次可直接使用 mode='json' 讀取此檔案。")
            except Exception as e:
                print(f"儲存 JSON 失敗: {e}")

        else:
            raise ValueError("請檢查輸入模式或路徑")

    def _interactive_select(self, img, win_name, num_points):
        """互動式選點 GUI (包含放大鏡功能與防呆)"""
        points = []
        img_disp = img.copy()
        
        # 定義滑鼠回調函數
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                points.append((x, y))
                # 畫點與標籤
                cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img_disp, f"{len(points)}", (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win_name, img_disp)
                print(f"[{win_name}] 已選取第 {len(points)}/{num_points} 點: ({x}, {y})")

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720) # 調整視窗大小以免圖片過大
        cv2.setMouseCallback(win_name, mouse_cb)
        
        print(f"\n請在『{win_name}』視窗中依序點選 {num_points} 個對應點。")
        print("按 'ESC' 鍵可提前結束 (若點數未滿將不會儲存)。")
        
        cv2.imshow(win_name, img_disp)
        
        while len(points) < num_points:
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC
                print("使用者取消選點。")
                break
        
        cv2.destroyWindow(win_name)
        return points

    def triangulate_points(self):
        """執行三角測量"""
        print("--- 執行線性三角測量 ---")
        self.raw_3d_points = {}
        sorted_keys = sorted(self.img_points.keys(), key=lambda x: int(x.replace("point", "")))
        
        for key in sorted_keys:
            pt = self.img_points[key]
            # 轉為 (1, 1, 2) 格式
            pt_L = np.array([[[pt['left'][0], pt['left'][1]]]], dtype=np.float32)
            pt_R = np.array([[[pt['right'][0], pt['right'][1]]]], dtype=np.float32)
            
            # 畸變校正 (保持在像素座標系 P=mtx)
            undist_L = cv2.undistortPoints(pt_L, self.mtxL, self.distL, P=self.mtxL)
            undist_R = cv2.undistortPoints(pt_R, self.mtxR, self.distR, P=self.mtxR)
            
            # 轉置為 2xN 格式
            pts_L_2xN = undist_L.reshape(-1, 2).T
            pts_R_2xN = undist_R.reshape(-1, 2).T
            
            # 三角測量 (4xN)
            points_4d = cv2.triangulatePoints(self.P_L, self.P_R, pts_L_2xN, pts_R_2xN)
            
            # 歸一化
            coord_3d = (points_4d[:3] / points_4d[3]).flatten()
            self.raw_3d_points[key] = coord_3d.tolist()
            
        print(f"已計算 {len(self.raw_3d_points)} 組原始 3D 座標")

    def load_vicon_data(self, csv_path):
        """讀取 VICON CSV 資料"""
        print(f"--- 載入 VICON 資料: {csv_path} ---")
        df = pd.read_csv(csv_path)
        self.vicon_3d_points = {}
        # 假設資料結構為 row 1 開始，每 3 欄為一個點的 X,Y,Z
        for i in range(15): 
            col_start = 2 + i * 3
            try:
                # 讀取數值並從 mm 轉為 m
                vals = df.iloc[1, col_start:col_start+3].values.astype(float) / 1000.0
                if np.isnan(vals).any(): continue
                self.vicon_3d_points[f"point{i+1}"] = vals.tolist()
            except IndexError:
                break
        print(f"已載入 {len(self.vicon_3d_points)} 組 VICON 座標")

    # ---------------- SVD 核心邏輯區塊 ----------------
    def _rigid_transform_3D(self, A, B):
        """
        SVD (Kabsch Algorithm) 計算剛體變換 B = R @ A + t
        Input: A, B 為 3xN 矩陣
        """
        assert A.shape == B.shape
        
        centroid_A = np.mean(A, axis=1, keepdims=True)
        centroid_B = np.mean(B, axis=1, keepdims=True)

        Am = A - centroid_A
        Bm = B - centroid_B
        H = Am @ Bm.T

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 修正反射矩陣情況
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t

    def align_coordinates_svd(self):
        """使用 SVD 將 Raw 3D Points 轉換至 VICON 座標系"""
        print("--- 執行 SVD 座標轉換 (Camera -> Vicon) ---")
        
        # 找出共同點
        common_keys = sorted([k for k in self.raw_3d_points.keys() if k in self.vicon_3d_points],
                             key=lambda x: int(x.replace("point", "")))
        
        if len(common_keys) < 3:
            raise ValueError("共同點過少，無法執行 SVD 轉換")

        # 構建 3xN 矩陣
        A = np.array([self.raw_3d_points[k] for k in common_keys]).T
        B = np.array([self.vicon_3d_points[k] for k in common_keys]).T
        
        # 計算轉換矩陣
        R_opt, t_opt = self._rigid_transform_3D(A, B)
        
        print("最佳旋轉矩陣 R:\n", R_opt)
        print("最佳平移向量 t:\n", t_opt.flatten())

        # 應用轉換到所有點
        self.aligned_points = {}
        for key, val in self.raw_3d_points.items():
            pt = np.array(val).reshape(3, 1)
            pt_new = R_opt @ pt + t_opt
            self.aligned_points[key] = pt_new.flatten().tolist()
            
        # 可選擇儲存轉換後的結果
        with open('final_transformed_points_svd.json', 'w') as f:
            json.dump(self.aligned_points, f, indent=4)
        print("座標轉換完成，已儲存至 final_transformed_points_svd.json")

    # ---------------- 誤差分析與繪圖 ----------------
    def compute_errors(self):
        """計算並顯示誤差"""
        print("\n--- 誤差分析 (Aligned vs Vicon) ---")
        errors_3d = []
        errors_2d = []
        
        keys = sorted(list(set(self.aligned_points.keys()) & set(self.vicon_3d_points.keys())),
                      key=lambda x: int(x.replace("point", "")))
        
        print(f"{'Point':<8} | {'3D Error (m)':<12} | {'2D(XY) Error (m)':<15}")
        print("-" * 45)
        
        for k in keys:
            calc = np.array(self.aligned_points[k])
            gt = np.array(self.vicon_3d_points[k])
            
            e3 = np.linalg.norm(calc - gt)
            e2 = np.linalg.norm(calc[:2] - gt[:2])
            
            errors_3d.append(e3)
            errors_2d.append(e2)
            print(f"{k:<8} | {e3:.4f}       | {e2:.4f}")
            
        print("-" * 45)
        print(f"平均 3D 誤差 (RMSE): {np.mean(errors_3d):.4f} m (Std: {np.std(errors_3d):.4f})")
        print(f"平均 2D 誤差 (RMSE): {np.mean(errors_2d):.4f} m (Std: {np.std(errors_2d):.4f})")

    def plot_2d_comparison(self):
        """繪製 2D 平面比較圖"""
        keys = sorted(list(set(self.aligned_points.keys()) & set(self.vicon_3d_points.keys())),
                      key=lambda x: int(x.replace("point", "")))
        
        calc_xy = np.array([self.aligned_points[k][:2] for k in keys])
        gt_xy = np.array([self.vicon_3d_points[k][:2] for k in keys])
        
        plt.figure(figsize=(10, 8))
        plt.scatter(calc_xy[:, 0], calc_xy[:, 1], c='red', marker='o', s=80, label='Calculated (SVD Aligned)')
        plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c='blue', marker='^', s=80, label='Vicon GT')
        
        for i, k in enumerate(keys):
            plt.plot([calc_xy[i, 0], gt_xy[i, 0]], [calc_xy[i, 1], gt_xy[i, 1]], 'gray', linestyle='--', alpha=0.5)
            plt.text(calc_xy[i, 0], calc_xy[i, 1], k, fontsize=9, color='darkred')
        
        plt.title("2D (X-Y) Coordinate Comparison after SVD Alignment")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    # 1. 設定檔案路徑 (請依實際情況修改)
    BASE_DIR = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate"
    PARAM_DIR = r"./20251125_exp1_雙目參數檔/exp1" 
    
    LEFT_NPZ = os.path.join(PARAM_DIR, "20251125_left_single_1_2.npz")
    RIGHT_NPZ = os.path.join(PARAM_DIR, "20251125_right_single_1_2.npz")
    STEREO_NPZ = os.path.join(PARAM_DIR, "stereo_rt_result_1_2.npz")
    
    # 影像路徑 (若使用手動選點)
    IMG_L = os.path.join(BASE_DIR, r"20251125_exp1_校正後圖片\court\origin_L.jpg")
    IMG_R = os.path.join(BASE_DIR, r"20251125_exp1_校正後圖片\court\origin_R.jpg")
    
    # 像素座標 JSON
    POINTS_JSON = os.path.join(BASE_DIR, "./origin_img/correspond_points_15_originimg_2.json")
    # VICON Ground Truth
    VICON_CSV = os.path.join(BASE_DIR, "court_1_vicon.csv")
    # 手動點選後儲存的 JSON
    NEW_JSON_NAME = "my_manual_points.json"

    try:
        # 2. 初始化系統與載入參數
        sys = StereoVisionSystem(LEFT_NPZ, RIGHT_NPZ, STEREO_NPZ)
        sys.load_parameters()

        # 3. [情況 1]：第一次執行，需要手動選點並存檔
        # sys.get_image_points(
        #     mode='manual', 
        #     left_img_path=IMG_L, 
        #     right_img_path=IMG_R, 
        #     num_points=15,
        #     save_path=NEW_JSON_NAME  # 指定儲存檔名
        # )
        
        # 3. [情況 2]：已經選過，直接讀取剛存好的檔案
        sys.get_image_points(mode='json', json_path=POINTS_JSON)
        
        # 4. 三角測量 (計算 Camera 座標系下的 Raw 3D Points)
        sys.triangulate_points()
        
        # 5. 載入 VICON 資料
        sys.load_vicon_data(VICON_CSV)
        
        # 6. SVD 座標轉換 (關鍵步驟：計算 R, t 並對齊座標)
        sys.align_coordinates_svd() 
        
        # 7. 誤差評估與繪圖 (使用對齊後的座標)
        sys.compute_errors()
        sys.plot_2d_comparison()
        
    except Exception as e:
        print(f"程式執行發生錯誤: {e}")
        import traceback
        traceback.print_exc()