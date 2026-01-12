"""
立體相機標定模組
提供單相機標定、雙相機標定、畸變校正和立體校正功能
"""

import numpy as np
import os
import cv2
import glob
from scipy.spatial.transform import Rotation as Rscipy
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# ============================================================================
# 配置類
# ============================================================================

@dataclass
class CalibrationConfig:
    """標定配置參數"""
    # 棋盤格參數
    chessboard_size: Tuple[int, int] = (9, 6)
    square_size: float = 0.1  # 單位：米
    
    # 單相機定參數
    batch_size: int = 50
    corner_subpix_win_size: Tuple[int, int] = (11, 11)
    corner_criteria_max_iter: int = 30
    corner_criteria_eps: float = 0.001
    
    # 雙目標定參數
    stereo_criteria_max_iter: int = 100
    stereo_criteria_eps: float = 0.0001
    
    # 路徑配置（可選，可在運行時設置）
    left_image_folder: Optional[str] = None
    right_image_folder: Optional[str] = None
    stereo_left_dir: Optional[str] = None
    stereo_right_dir: Optional[str] = None
    left_calibration_path: Optional[str] = None
    right_calibration_path: Optional[str] = None
    output_dir: str = '.'


# ============================================================================
# 單相機標定模組
# ============================================================================

class SingleCameraCalibrator:
    """單相機標定器"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            config.corner_criteria_max_iter,
            config.corner_criteria_eps
        )
    
    def _create_object_points(self) -> np.ndarray:
        """創建棋盤格世界座標點"""
        objp = np.zeros(
            (self.config.chessboard_size[0] * self.config.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0:self.config.chessboard_size[0],
            0:self.config.chessboard_size[1]
        ].T.reshape(-1, 2) * self.config.square_size
        return objp
    
    def _find_chessboard_corners(
        self,
        image_folder: str,
        show_corners: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """尋找棋盤格角點"""
        objp = self._create_object_points()
        obj_points = []
        img_points = []
        
        images = sorted(glob.glob(image_folder))
        if not images:
            raise ValueError(f"找不到圖片檔案: {image_folder}")
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"警告: 無法讀取圖片 {fname}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, self.config.chessboard_size, None
            )
            
            if ret:
                corners2 = cv2.cornerSubPix(
                    gray, corners,
                    self.config.corner_subpix_win_size,
                    (-1, -1),
                    self.criteria
                )
                obj_points.append(objp)
                img_points.append(corners2)
                
                if show_corners:
                    cv2.drawChessboardCorners(
                        img, self.config.chessboard_size, corners2, ret
                    )
                    cv2.imshow('Chessboard Corners', img)
                    cv2.waitKey(100)
        
        if show_corners:
            cv2.destroyAllWindows()
        
        return obj_points, img_points
    
    def _calculate_reprojection_errors(
        self,
        obj_points: List[np.ndarray],
        img_points: List[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> Dict:
        """計算重投影誤差"""
        total_error = 0
        per_view_errors = []
        
        for i in range(len(obj_points)):
            projected_points, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2).astype(np.float32)
            img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
            
            error = cv2.norm(img_pts, projected_points, cv2.NORM_L2) / len(projected_points)
            per_view_errors.append(error)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        
        return {
            'mean_reprojection_error': mean_error,
            'max_reprojection_error': max(per_view_errors),
            'min_reprojection_error': min(per_view_errors),
            'std_reprojection_error': np.std(per_view_errors),
            'per_view_errors': per_view_errors
        }
    
    def _analyze_camera_parameters(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        error_stats: Dict
    ) -> Dict:
        """分析相機參數"""
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        focal_ratio = fx / fy
        
        k1, k2, p1, p2, k3 = (
            dist_coeffs[0] if len(dist_coeffs[0]) >= 5
            else (*dist_coeffs[0], 0)
        )
        
        return {
            'focal_lengths': (fx, fy),
            'principal_point': (cx, cy),
            'focal_ratio': focal_ratio,
            'distortion_coeffs': {
                'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3
            }
        }
    
    def _print_calibration_results(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        error_stats: Dict,
        param_stats: Dict
    ):
        """打印標定結果"""
        print("\n相機內部參數矩陣:")
        print(camera_matrix)
        print("\n畸變係數:")
        print(dist_coeffs)
        
        print("\n=== 標定結果驗證 ===")
        print(f"平均重投影誤差: {error_stats['mean_reprojection_error']:.4f} 像素")
        print(f"最大單張圖片重投影誤差: {error_stats['max_reprojection_error']:.4f} 像素")
        print(f"最小單張圖片重投影誤差: {error_stats['min_reprojection_error']:.4f} 像素")
        print(f"重投影誤差標準差: {error_stats['std_reprojection_error']:.4f} 像素")
        
        fx, fy = param_stats['focal_lengths']
        cx, cy = param_stats['principal_point']
        print(f"\n焦距: fx={fx:.2f}, fy={fy:.2f}")
        print(f"主點: cx={cx:.2f}, cy={cy:.2f}")
        print(f"焦距比 (fx/fy): {param_stats['focal_ratio']:.4f} (理想值應接近1.0)")
        
        dist = param_stats['distortion_coeffs']
        print(f"\n畸變係數:")
        print(f"  徑向畸變 k1: {dist['k1']:.6f}")
        print(f"  徑向畸變 k2: {dist['k2']:.6f}")
        print(f"  切向畸變 p1: {dist['p1']:.6f}")
        print(f"  切向畸變 p2: {dist['p2']:.6f}")
        if dist['k3'] != 0:
            print(f"  徑向畸變 k3: {dist['k3']:.6f}")
        
        print(f"\n=== 標定品質評估 ===")
        mean_error = error_stats['mean_reprojection_error']
        if mean_error < 0.5:
            print("✓ 重投影誤差優秀 (< 0.5 像素)")
        elif mean_error < 1.0:
            print("✓ 重投影誤差良好 (< 1.0 像素)")
        elif mean_error < 2.0:
            print("⚠ 重投影誤差可接受 (< 2.0 像素)")
        else:
            print("✗ 重投影誤差過大 (≥ 2.0 像素)，建議重新標定")
        
        focal_ratio = param_stats['focal_ratio']
        if 0.95 < focal_ratio < 1.05:
            print("✓ 焦距比正常 (接近1.0)")
        else:
            print("⚠ 焦距比異常，可能表示相機參數有問題")
        
        if abs(dist['k1']) < 1.0 and abs(dist['k2']) < 1.0:
            print("✓ 徑向畸變係數正常")
        else:
            print("⚠ 徑向畸變係數過大，可能影響精度")
    
    def calibrate(
        self,
        image_folder: str,
        save_path: Optional[str] = None,
        show_corners: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List, List, Dict]]:
        """
        執行單相機標定
        
        Args:
            image_folder: 標定圖片資料夾路徑（支援 glob 模式）
            save_path: 保存路徑（.npz 格式）
            show_corners: 是否顯示角點檢測結果
        
        Returns:
            (camera_matrix, dist_coeffs, rvecs, tvecs, validation_results)
            如果失敗則返回 None
        """
        print("正在尋找棋盤格角點...")
        obj_points, img_points = self._find_chessboard_corners(
            image_folder, show_corners
        )
        
        if len(obj_points) == 0:
            print("錯誤: 找不到有效的棋盤格圖片")
            return None
        
        print(f"找到 {len(obj_points)} 張有效標定圖片")
        print("正在進行相機標定計算...")
        
        # 讀取第一張圖片以獲取影像尺寸
        sample_img = cv2.imread(sorted(glob.glob(image_folder))[0])
        gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        
        all_rvecs, all_tvecs = [], []
        camera_matrix, dist_coeffs = None, None
        
        # 分批處理以避免記憶體問題
        for i in range(0, len(obj_points), self.config.batch_size):
            try:
                batch_obj = obj_points[i:i+self.config.batch_size]
                batch_img = img_points[i:i+self.config.batch_size]
                
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    batch_obj, batch_img, image_size, None, None
                )
                all_rvecs.extend(rvecs)
                all_tvecs.extend(tvecs)
            except Exception as e:
                print(f"標定計算錯誤：{e}")
                return None
        
        # 計算驗證指標
        error_stats = self._calculate_reprojection_errors(
            obj_points, img_points, camera_matrix, dist_coeffs,
            all_rvecs, all_tvecs
        )
        
        param_stats = self._analyze_camera_parameters(
            camera_matrix, dist_coeffs, error_stats
        )
        
        # 打印結果
        self._print_calibration_results(
            camera_matrix, dist_coeffs, error_stats, param_stats
        )
        
        # 組合驗證結果
        validation_results = {
            **error_stats,
            **param_stats,
            'distortion_coeffs': list(param_stats['distortion_coeffs'].values())
        }
        
        # 保存結果
        if save_path:
            np.savez(
                save_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=all_rvecs,
                tvecs=all_tvecs,
                validation_results=validation_results
            )
            print(f"\n標定結果已保存至 {save_path}")
        
        print(f"包含 {len(obj_points)} 張有效標定圖片")
        
        return camera_matrix, dist_coeffs, all_rvecs, all_tvecs, validation_results


# ============================================================================
# 雙相機標定模組
# ============================================================================

class StereoCalibrator:
    """雙相機標定器"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            config.stereo_criteria_max_iter,
            config.stereo_criteria_eps
        )
    
    def calibrate(
        self,
        mtxL: np.ndarray,
        mtxR: np.ndarray,
        distL: np.ndarray,
        distR: np.ndarray,
        left_dir: str,
        right_dir: str,
        show_corners: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        執行雙相機標定
        
        Args:
            mtxL, mtxR: 左右相機內部參數矩陣
            distL, distR: 左右相機畸變係數
            left_dir, right_dir: 左右相機標定圖片資料夾
            show_corners: 是否顯示角點檢測結果
        
        Returns:
            (R, T) 旋轉矩陣和平移向量
        """
        objp = np.zeros(
            (self.config.chessboard_size[0] * self.config.chessboard_size[1], 3),
            np.float32
        )
        objp[:, :2] = np.mgrid[
            0:self.config.chessboard_size[0],
            0:self.config.chessboard_size[1]
        ].T.reshape(-1, 2) * self.config.square_size
        
        left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')))
        right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')))
        
        if len(left_images) != len(right_images):
            raise ValueError("左右相機圖片數量不一致")
        
        imgpoints_left = []
        imgpoints_right = []
        objpoints = []
        image_size = None
        
        for frameL, frameR in zip(left_images, right_images):
            img_left = cv2.imread(frameL)
            img_right = cv2.imread(frameR)
            
            if img_left is None or img_right is None:
                print(f"警告: 無法讀取圖片 {frameL} 或 {frameR}")
                continue
            
            grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                if grayL.shape[::-1] == grayR.shape[::-1]:
                    image_size = grayL.shape[::-1]
                else:
                    raise ValueError("左右相機影像尺寸不一致")
            
            retL, cornersL = cv2.findChessboardCorners(grayL, self.config.chessboard_size)
            retR, cornersR = cv2.findChessboardCorners(grayR, self.config.chessboard_size)
            
            if retL and retR:
                cornersL = cv2.cornerSubPix(
                    grayL, cornersL,
                    self.config.corner_subpix_win_size,
                    (-1, -1),
                    self.criteria
                )
                cornersR = cv2.cornerSubPix(
                    grayR, cornersR,
                    self.config.corner_subpix_win_size,
                    (-1, -1),
                    self.criteria
                )
                
                if show_corners:
                    cv2.drawChessboardCorners(img_left, self.config.chessboard_size, cornersL, retL)
                    cv2.imshow('img_left', img_left)
                    cv2.drawChessboardCorners(img_right, self.config.chessboard_size, cornersR, retR)
                    cv2.imshow('img_right', img_right)
                    cv2.waitKey(500)
                
                objpoints.append(objp)
                imgpoints_left.append(cornersL)
                imgpoints_right.append(cornersR)
        
        if show_corners:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        
        if len(objpoints) == 0:
            raise ValueError("找不到有效的立體標定圖片對")
        
        # 固定內參，僅估計 R, T 外參
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtxL, distL, mtxR, distR,
            image_size, criteria=self.criteria, flags=flags
        )
        
        print(f"image_size: {image_size}")
        print(f"平均重投影誤差: {ret:.6f} 像素")
        
        return R, T


# ============================================================================
# 畸變校正模組
# ============================================================================

class UndistortionProcessor:
    """畸變校正處理器"""
    
    @staticmethod
    def undistort(
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image: np.ndarray,
        crop: bool = True
    ) -> np.ndarray:
        """
        校正影像畸變
        
        Args:
            camera_matrix: 相機內部參數矩陣
            dist_coeffs: 畸變係數
            image: 待校正影像
            crop: 是否裁剪 ROI
        
        Returns:
            校正後的影像
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h)
        )
        
        dst = cv2.undistort(
            image, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        
        if crop and roi is not None and all(roi):
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        
        return dst


# ============================================================================
# 立體校正模組
# ============================================================================

class StereoRectifier:
    """立體校正器"""
    
    @staticmethod
    def compute_rectification(
        mtxL: np.ndarray,
        mtxR: np.ndarray,
        distL: np.ndarray,
        distR: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: Tuple[int, int],
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple, Tuple]:
        """
        計算立體校正參數
        
        Returns:
            (R1, R2, P1, P2, Q, roi1, roi2)
        """
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtxL, distL, mtxR, distR,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=1
        )
        return R1, R2, P1, P2, Q, roi1, roi2
    
    @staticmethod
    def compute_rectification_maps(
        mtxL: np.ndarray,
        mtxR: np.ndarray,
        distL: np.ndarray,
        distR: np.ndarray,
        R1: np.ndarray,
        R2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        計算校正映射表
        
        Returns:
            (left_map_1, left_map_2, right_map_1, right_map_2)
        """
        left_map_1, left_map_2 = cv2.initUndistortRectifyMap(
            mtxL, distL, R1, P1, image_size, cv2.CV_16SC2
        )
        right_map_1, right_map_2 = cv2.initUndistortRectifyMap(
            mtxR, distR, R2, P2, image_size, cv2.CV_16SC2
        )
        return left_map_1, left_map_2, right_map_1, right_map_2
    
    @staticmethod
    def rectify_images(
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_map_1: np.ndarray,
        left_map_2: np.ndarray,
        right_map_1: np.ndarray,
        right_map_2: np.ndarray,
        roi1: Tuple,
        roi2: Tuple
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        校正立體影像對
        
        Returns:
            (left_rectified, right_rectified, stacked_img)
        """
        left_rectified = cv2.remap(
            left_img, left_map_1, left_map_2, cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            right_img, right_map_1, right_map_2, cv2.INTER_LINEAR
        )
        
        # 計算共同 ROI
        x1, y1, w1, h1 = roi1
        x2, y2, w2, h2 = roi2
        x2_end = x2 + w2
        
        common_w = min(w1, w2)
        common_h = min(h1, h2)
        
        if common_w > 0 and common_h > 0:
            left_rect_cropped = left_rectified[y1:y1+common_h, x1:x1+common_w]
            right_rect_cropped = right_rectified[y2:y2+common_h, (x2_end-common_w):x2_end]
            stacked_img = np.hstack((left_rect_cropped, right_rect_cropped))
        else:
            print("警告: ROI 沒有交集，使用完整影像")
            left_rect_cropped = left_rectified
            right_rect_cropped = right_rectified
            stacked_img = np.hstack((left_rectified, right_rectified))
        
        return left_rect_cropped, right_rect_cropped, stacked_img


# ============================================================================
# 工具函數
# ============================================================================

def load_calibration_params(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """載入相機標定參數"""
    data = np.load(npz_path, allow_pickle=True)
    return data['camera_matrix'], data['dist_coeffs']


def load_validation_results(npz_path: str) -> Optional[Dict]:
    """載入並顯示驗證結果"""
    data = np.load(npz_path, allow_pickle=True)
    
    if 'validation_results' in data:
        validation_results = data['validation_results'].item()
        print(f"\n從 {npz_path} 讀取驗證結果:")
        print(f"平均重投影誤差: {validation_results['mean_reprojection_error']:.4f} 像素")
        print(f"最大重投影誤差: {validation_results['max_reprojection_error']:.4f} 像素")
        print(f"最小重投影誤差: {validation_results['min_reprojection_error']:.4f} 像素")
        print(f"標準差: {validation_results['std_reprojection_error']:.4f} 像素")
        
        if 'focal_lengths' in validation_results:
            fx, fy = validation_results['focal_lengths']
            cx, cy = validation_results['principal_point']
            print(f"焦距: fx={fx:.2f}, fy={fy:.2f}")
            print(f"主點: cx={cx:.2f}, cy={cy:.2f}")
            print(f"焦距比: {validation_results['focal_ratio']:.4f}")
        
        return validation_results
    else:
        print(f"文件 {npz_path} 中沒有驗證結果")
        return None


def compute_rotation_angles(R: np.ndarray) -> np.ndarray:
    """從旋轉矩陣計算歐拉角（度）"""
    rot = Rscipy.from_matrix(R)
    return rot.as_euler('xyz', degrees=True)


def compute_baseline(T: np.ndarray) -> float:
    """計算基線長度（米）"""
    # 使用 np.linalg.norm 避免 NumPy 1.25+ 的 deprecation warning
    return float(np.linalg.norm(T))


def visualize_rectified_images(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    stacked_img: np.ndarray
):
    """視覺化校正後的影像"""
    plt.figure(figsize=(24, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB))
    plt.title("Left Rectified")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB))
    plt.title("Right Rectified")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(stacked_img, cv2.COLOR_BGR2RGB))
    plt.title("Stacked Rectified")
    plt.axis('off')
    
    origin_stacked = np.hstack((left_img, right_img))
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(origin_stacked, cv2.COLOR_BGR2RGB))
    plt.title("Original Stacked")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 主程式
# ============================================================================

def main():
    """主執行函數"""
    # 配置參數
    config = CalibrationConfig(
        chessboard_size=(9, 6),
        square_size=0.1,
        batch_size=15,
        # 設定路徑（請根據實際情況修改）
        left_image_folder=r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\left\selected_15\*.jpg",
        right_image_folder=r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\right\selected_15\*.jpg",
        stereo_left_dir=r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\left\stereo_image_15",
        stereo_right_dir=r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\right\stereo_image_15",
        left_calibration_path='20250924_left_validation.npz',
        right_calibration_path='20250924_right_validation.npz',
        output_dir='.'
    )
    
    # ========================================================================
    # 步驟 1: 單相機標定
    # ========================================================================
    print("=" * 60)
    print("步驟 1: 單相機標定")
    print("=" * 60)
    
    calibrator = SingleCameraCalibrator(config)
    
    # 左相機標定
    print("\n=== 左相機標定 (含驗證) ===")
    left_result = calibrator.calibrate(
        image_folder=config.left_image_folder,
        save_path=config.left_calibration_path,
        show_corners=False
    )
    
    # 右相機標定
    print("\n=== 右相機標定 (含驗證) ===")
    right_result = calibrator.calibrate(
        image_folder=config.right_image_folder,
        save_path=config.right_calibration_path,
        show_corners=False
    )
    
    if left_result is None or right_result is None:
        print("錯誤: 單相機標定失敗")
        return
    
    # ========================================================================
    # 步驟 2: 載入標定參數
    # ========================================================================
    print("\n" + "=" * 60)
    print("步驟 2: 載入標定參數")
    print("=" * 60)
    
    mtxL, distL = load_calibration_params(config.left_calibration_path)
    mtxR, distR = load_calibration_params(config.right_calibration_path)
    
    print("\n左相機參數:")
    print(mtxL)
    print(distL)
    print("\n右相機參數:")
    print(mtxR)
    print(distR)
    
    # 讀取驗證結果
    print("\n=== 讀取左相機驗證結果 ===")
    load_validation_results(config.left_calibration_path)
    
    print("\n=== 讀取右相機驗證結果 ===")
    load_validation_results(config.right_calibration_path)
    
    # ========================================================================
    # 步驟 3: 雙相機標定
    # ========================================================================
    print("\n" + "=" * 60)
    print("步驟 3: 雙相機標定")
    print("=" * 60)
    
    stereo_calibrator = StereoCalibrator(config)
    R, T = stereo_calibrator.calibrate(
        mtxL, mtxR, distL, distR,
        config.stereo_left_dir,
        config.stereo_right_dir,
        show_corners=False
    )
    
    # 計算 baseline 和旋轉角度
    baseline = compute_baseline(T)
    angles = compute_rotation_angles(R)
    
    print(f"\nR matrix:\n{R}")
    print(f"\nT vector:\n{T}")
    print(f"\nbaseline: {baseline:.4f} m")
    print(f"相機沿著X軸旋轉: {angles[0]:.2f} 度")
    print(f"相機沿著Y軸旋轉: {angles[1]:.2f} 度")
    print(f"相機沿著Z軸旋轉: {angles[2]:.2f} 度")
    
    # ========================================================================
    # 步驟 4: 立體校正
    # ========================================================================
    print("\n" + "=" * 60)
    print("步驟 4: 立體校正")
    print("=" * 60)
    
    # 讀取測試影像以獲取尺寸
    test_left_path = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\left\best_15_selected\best_image_01.jpg"
    test_right_path = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20250924_chessboard_img\dual\right\best_15_selected\best_image_01.jpg"
    
    left_img = cv2.imread(test_left_path)
    right_img = cv2.imread(test_right_path)
    
    if left_img is None or right_img is None:
        print("錯誤: 無法讀取測試影像")
        return
    
    image_size = (left_img.shape[1], left_img.shape[0])
    print(f"影像尺寸: {image_size}")
    
    # 計算校正參數
    R1, R2, P1, P2, Q, roi1, roi2 = StereoRectifier.compute_rectification(
        mtxL, mtxR, distL, distR, R, T, image_size
    )
    print(f"roi1 : {roi1}\n")
    print(f"roi2 : {roi2}\n")
    
    # 計算映射表
    left_map_1, left_map_2, right_map_1, right_map_2 = (
        StereoRectifier.compute_rectification_maps(
            mtxL, mtxR, distL, distR, R1, R2, P1, P2, image_size
        )
    )
    
    # 校正影像
    left_rect, right_rect, stacked_img = StereoRectifier.rectify_images(
        left_img, right_img,
        left_map_1, left_map_2, right_map_1, right_map_2, roi1, roi2
    )
    
    print(f"裁切後左影像尺寸: {left_rect.shape}")
    print(f"裁切後右影像尺寸: {right_rect.shape}")
    
    # 視覺化結果
    visualize_rectified_images(
        left_img, right_img, left_rect, right_rect, stacked_img
    )
    
    print("\n標定流程完成！")


if __name__ == "__main__":
    main()
