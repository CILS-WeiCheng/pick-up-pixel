'''
先對整張圖做畸變校正，再進行三角測量
'''

import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

# 常數定義
ESC_KEY = 27
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3
VICON_MM_TO_M = 1000.0


class StereoVisionSystem:
    """雙目視覺系統，用於立體視覺重建與座標轉換"""
    
    def __init__(self, left_calib_path: str, right_calib_path: str, stereo_calib_path: str):
        """
        初始化雙目視覺系統
        
        Args:
            left_calib_path: 左相機校正參數檔案路徑
            right_calib_path: 右相機校正參數檔案路徑
            stereo_calib_path: 雙目校正參數檔案路徑
        """
        self.left_calib_path = left_calib_path
        self.right_calib_path = right_calib_path
        self.stereo_calib_path = stereo_calib_path
        
        # 參數容器
        self.mtxL: Optional[np.ndarray] = None
        self.distL: Optional[np.ndarray] = None
        self.mtxR: Optional[np.ndarray] = None
        self.distR: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None
        self.P_L: Optional[np.ndarray] = None
        self.P_R: Optional[np.ndarray] = None
        
        # 優化的新相機矩陣（用於最大化有效像素區域）
        self.new_mtxL: Optional[np.ndarray] = None
        self.new_mtxR: Optional[np.ndarray] = None
        
        # 資料容器
        self.img_points: Dict[str, Dict[str, List[int]]] = {}       # 像素座標
        self.raw_3d_points: Dict[str, List[float]] = {}             # 三角測量後的原始 3D 座標 (左相機座標系)
        self.vicon_3d_points: Dict[str, List[float]] = {}           # VICON Ground Truth
        self.aligned_points: Dict[str, List[float]] = {}            # 經 SVD 轉換後對齊的 3D 座標
    
    @staticmethod
    def _sort_point_keys(keys: List[str]) -> List[str]:
        """將點名稱按數字順序排序"""
        return sorted(keys, key=lambda x: int(x.replace("point", "")))

    @staticmethod
    def imread_unicode(file_path: str) -> np.ndarray:
        """
        讀取包含中文路徑的圖片
        
        Args:
            file_path: 圖片檔案路徑
            
        Returns:
            圖片陣列
            
        Raises:
            FileNotFoundError: 無法讀取圖片時拋出
        """
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取圖片: {file_path}")
        return img

    def load_parameters(self, image_size: Optional[Tuple[int, int]] = None) -> None:
        """
        載入單目與雙目參數並建構投影矩陣
        
        Args:
            image_size: 圖像尺寸 (width, height)，若提供則計算優化的新相機矩陣
        """
        print("--- 載入校正參數 ---")
        
        # 載入單目參數
        left_data = np.load(self.left_calib_path, allow_pickle=True)
        self.mtxL = left_data['camera_matrix']
        self.distL = left_data['dist_coeffs']
        
        right_data = np.load(self.right_calib_path, allow_pickle=True)
        self.mtxR = right_data['camera_matrix']
        self.distR = right_data['dist_coeffs']
        
        # 載入雙目外參
        stereo_data = np.load(self.stereo_calib_path, allow_pickle=True)
        self.R = stereo_data['R']
        self.T = stereo_data['T']
        
        print(f"Baseline: {np.linalg.norm(self.T):.4f} m")

        # 如果提供了圖像尺寸，計算優化的新相機矩陣
        if image_size is not None:
            w, h = image_size
            print(f"計算優化的新相機矩陣 (圖像尺寸: {w}x{h})...")
            self.new_mtxL, _ = cv2.getOptimalNewCameraMatrix(
                self.mtxL, self.distL, (w, h), 0, (w, h)
            )
            self.new_mtxR, _ = cv2.getOptimalNewCameraMatrix(
                self.mtxR, self.distR, (w, h), 0, (w, h)
            )
            print("優化的新相機矩陣計算完成。")
            # 使用新相機矩陣建構投影矩陣
            mtxL_used = self.new_mtxL
            mtxR_used = self.new_mtxR
        else:
            # 使用原始相機矩陣
            mtxL_used = self.mtxL
            mtxR_used = self.mtxR
            print("使用原始相機矩陣建構投影矩陣。")

        # 建構投影矩陣 P = K @ [R|T]
        # 左相機為世界座標原點 [I|0]
        self.P_L = mtxL_used @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # 右相機相對位置 [R|T]
        self.P_R = mtxR_used @ np.hstack((self.R, self.T))
        
        print("投影矩陣建構完成。")

    def get_image_points(
        self,
        mode: str = 'json',
        json_path: Optional[str] = None,
        left_img_path: Optional[str] = None,
        right_img_path: Optional[str] = None,
        num_points: int = 15,
        save_path: str = "manual_points.json"
    ) -> None:
        """
        獲取圖像座標點
        
        Args:
            mode: 'json' (讀檔) 或 'manual' (手動選點)
            json_path: JSON 檔案路徑 (mode='json' 時必填)
            left_img_path: 左圖路徑 (mode='manual' 時必填)
            right_img_path: 右圖路徑 (mode='manual' 時必填)
            num_points: 手動選點時目標點數
            save_path: 手動選點後要儲存的 JSON 檔名
        """
        if mode == 'json':
            if not json_path:
                raise ValueError("json_path 參數為必填")
            self._load_points_from_json(json_path)
        
        elif mode == 'manual':
            if not (left_img_path and right_img_path):
                raise ValueError("left_img_path 和 right_img_path 參數為必填")
            self._manual_select_points(left_img_path, right_img_path, num_points, save_path)
        
        else:
            raise ValueError(f"不支援的模式: {mode}，請使用 'json' 或 'manual'")
    
    def _load_points_from_json(self, json_path: str) -> None:
        """從 JSON 檔案載入點位"""
        print(f"--- 從 JSON 載入點位: {json_path} ---")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.img_points = json.load(f)
        
        sorted_keys = self._sort_point_keys(list(self.img_points.keys()))
        print(f"已載入 {len(self.img_points)} 個點: {', '.join(sorted_keys)}")
    
    def _manual_select_points(
        self,
        left_img_path: str,
        right_img_path: str,
        num_points: int,
        save_path: str
    ) -> None:
        """手動選點模式（先對整張圖做畸變校正，再選點）"""
        print(f"--- 啟動手動選點模式 (目標: {num_points} 點) ---")
        
        # 讀取原始圖片
        img_L_raw = self.imread_unicode(left_img_path)
        img_R_raw = self.imread_unicode(right_img_path)
        
        # 獲取圖像尺寸
        h, w = img_L_raw.shape[:2]
        image_size = (w, h)
        
        # 如果尚未計算優化的新相機矩陣，則計算
        if self.new_mtxL is None or self.new_mtxR is None:
            print(f"計算優化的新相機矩陣 (圖像尺寸: {w}x{h})...")
            self.new_mtxL, _ = cv2.getOptimalNewCameraMatrix(
                self.mtxL, self.distL, image_size, 0, image_size
            )
            self.new_mtxR, _ = cv2.getOptimalNewCameraMatrix(
                self.mtxR, self.distR, image_size, 0, image_size
            )
            print("優化的新相機矩陣計算完成。")
            
            # 使用新相機矩陣重新建構投影矩陣
            self.P_L = self.new_mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
            self.P_R = self.new_mtxR @ np.hstack((self.R, self.T))
            print("投影矩陣已更新為使用優化的新相機矩陣。")
        
        # 使用優化的新相機矩陣對整張圖做畸變校正
        print("正在對左右圖進行畸變校正（使用優化的新相機矩陣）...")
        img_L = cv2.undistort(img_L_raw, self.mtxL, self.distL, None, self.new_mtxL)
        img_R = cv2.undistort(img_R_raw, self.mtxR, self.distR, None, self.new_mtxR)
        print("畸變校正完成，請在校正後的圖上選點。")
        
        # 在校正後的圖上選點
        pts_L = self._interactive_select(img_L, "Select Left Image (Undistorted)", num_points)
        pts_R = self._interactive_select(img_R, "Select Right Image (Undistorted)", num_points)
        
        if len(pts_L) != num_points or len(pts_R) != num_points:
            print(f"警告: 選取的點數不足 (左: {len(pts_L)}, 右: {len(pts_R)})，請重新執行。")
            return

        # 建構符合格式的字典: {"point1": {"left": [u, v], "right": [u, v]}, ...}
        self.img_points = {
            f"point{i+1}": {
                "left": [int(pts_L[i][0]), int(pts_L[i][1])],
                "right": [int(pts_R[i][0]), int(pts_R[i][1])]
            }
            for i in range(num_points)
        }
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.img_points, f, indent=4, ensure_ascii=False)
            print(f"選點完成！座標已儲存至: {save_path}")
            print("格式已對齊，下次可直接使用 mode='json' 讀取此檔案。")
        except Exception as e:
            print(f"儲存 JSON 失敗: {e}")

    def _interactive_select(self, img: np.ndarray, win_name: str, num_points: int) -> List[Tuple[int, int]]:
        """
        互動式選點 GUI
        
        Args:
            img: 輸入圖片
            win_name: 視窗名稱
            num_points: 目標選點數量
            
        Returns:
            選取的點座標列表
        """
        points: List[Tuple[int, int]] = []
        img_disp = img.copy()
        
        def mouse_cb(event: int, x: int, y: int, flags: int, param) -> None:
            """滑鼠回調函數"""
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                points.append((x, y))
                cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img_disp, f"{len(points)}", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win_name, img_disp)
                print(f"[{win_name}] 已選取第 {len(points)}/{num_points} 點: ({x}, {y})")

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(win_name, mouse_cb)
        
        print(f"\n請在『{win_name}』視窗中依序點選 {num_points} 個對應點。")
        print("按 'ESC' 鍵可提前結束 (若點數未滿將不會儲存)。")
        
        cv2.imshow(win_name, img_disp)
        
        while len(points) < num_points:
            key = cv2.waitKey(10) & 0xFF
            if key == ESC_KEY:
                print("使用者取消選點。")
                break
        
        cv2.destroyWindow(win_name)
        return points

    def triangulate_points(self, points_are_undistorted: bool = True) -> None:
        """
        執行三角測量，計算原始 3D 座標（左相機座標系）
        
        Args:
            points_are_undistorted: 若為 True，表示點已經在校正後的圖上選取，不需要再做畸變校正
        """
        print("--- 執行線性三角測量 ---")
        self.raw_3d_points = {}
        sorted_keys = self._sort_point_keys(list(self.img_points.keys()))
        
        for key in sorted_keys:
            pt = self.img_points[key]
            
            if points_are_undistorted:
                # 點已經在校正後的圖上選取，直接使用像素座標
                pts_L_2xN = np.array([[pt['left'][0]], [pt['left'][1]]], dtype=np.float32)
                pts_R_2xN = np.array([[pt['right'][0]], [pt['right'][1]]], dtype=np.float32)
            else:
                # 點在原圖上選取，需要做單點畸變校正
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

    def load_vicon_data(self, csv_path: str, expected_points: Optional[List[str]] = None) -> None:
        """
        讀取 VICON CSV 資料
        
        Args:
            csv_path: VICON CSV 檔案路徑
            expected_points: 預期要讀取的點列表，若為 None 則根據 self.img_points 中存在的點來讀取
        """
        print(f"--- 載入 VICON 資料: {csv_path} ---")
        df = pd.read_csv(csv_path)
        self.vicon_3d_points = {}
        
        # 決定要讀取哪些點
        point_keys = self._determine_point_keys(expected_points)
        
        # 讀取對應的 VICON 資料
        loaded_count = 0
        for point_key in point_keys:
            try:
                point_idx = int(point_key.replace("point", "")) - 1
                col_start = 2 + point_idx * 3
                
                # 讀取數值並從 mm 轉為 m
                vals = df.iloc[1, col_start:col_start+3].values.astype(float) / VICON_MM_TO_M
                if np.isnan(vals).any():
                    print(f"警告: {point_key} 的 VICON 資料包含 NaN，跳過")
                    continue
                self.vicon_3d_points[point_key] = vals.tolist()
                loaded_count += 1
            except (IndexError, ValueError) as e:
                print(f"警告: 無法讀取 {point_key} 的 VICON 資料: {e}")
                continue
        
        print(f"已載入 {loaded_count} 組 VICON 座標")
        if loaded_count < len(point_keys):
            print(f"注意: 僅成功載入 {loaded_count}/{len(point_keys)} 個點的 VICON 資料")
    
    def _determine_point_keys(self, expected_points: Optional[List[str]]) -> List[str]:
        """決定要讀取的點鍵值列表"""
        if expected_points is not None:
            return expected_points
        
        if not hasattr(self, 'img_points') or len(self.img_points) == 0:
            print("警告: img_points 尚未載入，將嘗試讀取所有可能的點（最多15個）")
            return [f"point{i+1}" for i in range(15)]
        
        point_keys = self._sort_point_keys(list(self.img_points.keys()))
        print(f"根據 img_points，將讀取 {len(point_keys)} 個對應的 VICON 點")
        return point_keys

    def _rigid_transform_3D(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SVD (Kabsch Algorithm) 計算剛體變換 B = R @ A + t
        
        Args:
            A: 源座標點 3xN 矩陣
            B: 目標座標點 3xN 矩陣
            
        Returns:
            旋轉矩陣 R 和平移向量 t
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

    def align_coordinates_svd(self, output_path: str = 'final_transformed_points_svd.json') -> None:
        """
        使用 SVD 將 Raw 3D Points 轉換至 VICON 座標系
        
        Args:
            output_path: 轉換後結果的儲存路徑
        """
        print("--- 執行 SVD 座標轉換 (Camera -> Vicon) ---")
        
        # 找出共同點
        common_keys = self._sort_point_keys([
            k for k in self.raw_3d_points.keys() if k in self.vicon_3d_points
        ])
        
        if len(common_keys) < MIN_POINTS_FOR_SVD:
            raise ValueError(f"共同點過少 ({len(common_keys)} < {MIN_POINTS_FOR_SVD})，無法執行 SVD 轉換")

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
            
        # 儲存轉換後的結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.aligned_points, f, indent=4, ensure_ascii=False)
        print(f"座標轉換完成，已儲存至 {output_path}")

    def _get_common_keys(self) -> List[str]:
        """取得對齊點與 VICON 點的共同鍵值列表"""
        common_keys = set(self.aligned_points.keys()) & set(self.vicon_3d_points.keys())
        return self._sort_point_keys(list(common_keys))
    
    def compute_errors(self) -> None:
        """計算並顯示誤差"""
        print("\n--- 誤差分析 (Aligned vs Vicon) ---")
        errors_3d: List[float] = []
        errors_2d: List[float] = []
        
        keys = self._get_common_keys()
        
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

    def plot_2d_comparison(self) -> None:
        """繪製 2D 平面比較圖"""
        keys = self._get_common_keys()
        
        calc_xy = np.array([self.aligned_points[k][:2] for k in keys])
        gt_xy = np.array([self.vicon_3d_points[k][:2] for k in keys])
        
        plt.figure(figsize=(10, 8))
        plt.scatter(calc_xy[:, 0], calc_xy[:, 1], c='red', marker='o', s=80,
                    label='Calculated (SVD Aligned)')
        plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c='blue', marker='^', s=80, label='Vicon GT')
        
        for i, k in enumerate(keys):
            plt.plot([calc_xy[i, 0], gt_xy[i, 0]], [calc_xy[i, 1], gt_xy[i, 1]],
                     'gray', linestyle='--', alpha=0.5)
            plt.text(calc_xy[i, 0], calc_xy[i, 1], k, fontsize=9, color='darkred')
        
        plt.title("2D (X-Y) Coordinate Comparison after SVD Alignment")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def main():
    """主程式執行函數"""
    # 設定檔案路徑 (請依實際情況修改)
    BASE_DIR = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate"
    PARAM_DIR = r"./20251125_exp1_雙目參數檔/exp1"
    
    LEFT_NPZ = os.path.join(PARAM_DIR, "20251125_left_single_1_2.npz")
    RIGHT_NPZ = os.path.join(PARAM_DIR, "20251125_right_single_1_2.npz")
    STEREO_NPZ = os.path.join(PARAM_DIR, "stereo_rt_result_1_2.npz")
    
    # 影像路徑 (若使用手動選點)
    IMG_L = os.path.join(BASE_DIR, r"20251125_exp1_校正後圖片\court\origin_L.jpg")
    IMG_R = os.path.join(BASE_DIR, r"20251125_exp1_校正後圖片\court\origin_R.jpg")
    
    # 像素座標 JSON
    POINTS_JSON = os.path.join(BASE_DIR, "vicon_pixel.json")
    # POINTS_JSON = os.path.join(BASE_DIR, "./origin_img/correspond_points_15_originimg_2.json")
    
    # VICON Ground Truth
    VICON_CSV = os.path.join(BASE_DIR, "court_1_vicon.csv")
    
    # 手動點選後儲存的 JSON
    NEW_JSON_NAME = "my_manual_points.json"

    try:
        # 初始化系統與載入參數
        sys = StereoVisionSystem(LEFT_NPZ, RIGHT_NPZ, STEREO_NPZ)
        
        # [情況 1]：第一次執行，需要手動選點並存檔
        # 在 manual 模式下，會自動計算優化的新相機矩陣
        # sys.load_parameters()  # 先載入基本參數
        
        # sys.get_image_points(
        #     mode='manual',
        #     left_img_path=IMG_L,
        #     right_img_path=IMG_R,
        #     num_points=15,
        #     save_path=NEW_JSON_NAME
        # )
        
        # [情況 2]：已經選過，直接讀取剛存好的檔案
        # 如果 JSON 中的點是在使用優化新相機矩陣校正後的圖上選的，
        # 需要確保投影矩陣也使用新相機矩陣
        sys.load_parameters()  # 使用原始相機矩陣
        # 或者指定圖像尺寸來計算優化的新相機矩陣：
        img_sample = sys.imread_unicode(IMG_L)
        h, w = img_sample.shape[:2]
        sys.load_parameters(image_size=(w, h))  # 使用優化的新相機矩陣
        sys.get_image_points(mode='json', json_path=POINTS_JSON)
        
        # 三角測量 (計算 Camera 座標系下的 Raw 3D Points)
        # points_are_undistorted=True 表示點已經在校正後的圖上選取
        sys.triangulate_points(points_are_undistorted=True)
        
        # 載入 VICON 資料
        sys.load_vicon_data(VICON_CSV)
        
        # SVD 座標轉換 (關鍵步驟：計算 R, t 並對齊座標)
        sys.align_coordinates_svd()
        
        # 誤差評估與繪圖 (使用對齊後的座標)
        sys.compute_errors()
        sys.plot_2d_comparison()
        
    except Exception as e:
        print(f"程式執行發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()