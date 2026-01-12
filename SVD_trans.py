import numpy as np
import cv2
import glob
import os
import json
from scipy.spatial.transform import Rotation as Rscipy

def imread_unicode(file_path):
    """讀取包含中文路徑的圖片"""
    img_array = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def calibration_single_camera(image_folder, chessboard_size=(9, 6), square_size=0.1, save_path=None):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points, img_points = [], []
    images = glob.glob(image_folder)
    
    if not images:
        return None, None

    img_shape = None
    for fname in images:
        img = imread_unicode(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

    if not obj_points:
        print(f"未檢測到棋盤格: {image_folder}")
        return None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
    
    if save_path:
        np.savez(save_path, camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"單目標定 RMSE: {ret:.4f} (已儲存至 {save_path})")

    return mtx, dist

def stereo_calibration(mtxL, distL, mtxR, distR, left_dir, right_dir, chessboard_size=(9, 6), square_size=0.1):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')))

    objpoints, imgpoints_left, imgpoints_right = [], [], []
    image_size = None

    for frameL, frameR in zip(left_images, right_images):
        imgL = imread_unicode(frameL)
        imgR = imread_unicode(frameR)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = grayL.shape[::-1]

        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtxL_opt, distL_opt, mtxR_opt, distR_opt, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        image_size, criteria=criteria, flags=flags
    )
    
    print(f"雙目 RMSE: {ret:.4f}")
    return mtxL_opt, distL_opt, mtxR_opt, distR_opt, R, T

class StereoReconstructor:
    def __init__(self, mtxL, distL, mtxR, distR, R, T, real_baseline=None):
        self.mtxL, self.distL = mtxL, distL
        self.mtxR, self.distR = mtxR, distR
        self.R = R
        
        if real_baseline is not None:
            self.T = T * (real_baseline / np.linalg.norm(T))
        else:
            self.T = T

        self.P_L = mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_R = mtxR @ np.hstack((R, self.T))

    def calculate_3d_point(self, u_L, v_L, u_R, v_R):
        """輸入左右圖像素座標，輸出左相機座標系下的 3D 座標 (X, Y, Z)"""
        pt_L = np.array([[[u_L, v_L]]], dtype=np.float32)
        pt_R = np.array([[[u_R, v_R]]], dtype=np.float32)

        undist_pts_L = cv2.undistortPoints(pt_L, self.mtxL, self.distL, P=self.mtxL)
        undist_pts_R = cv2.undistortPoints(pt_R, self.mtxR, self.distR, P=self.mtxR)

        # 轉置為 2xN 格式以符合 triangulatePoints 需求
        pts_L_2xN = undist_pts_L.reshape(-1, 2).T
        pts_R_2xN = undist_pts_R.reshape(-1, 2).T
        
        points_4d = cv2.triangulatePoints(self.P_L, self.P_R, pts_L_2xN, pts_R_2xN)
        
        # 歸一化
        w = points_4d[3]
        return (points_4d[:3] / w).flatten()

def rigid_transform_3D(A, B):
    """SVD (Kabsch Algorithm) 計算剛體變換 B = R @ A + t"""
    assert A.shape == B.shape
    
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ Bm.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

def apply_transform(points_dict, R, t):
    transformed_dict = {}
    for key, val in points_dict.items():
        pt = np.array(val).reshape(3, 1)
        pt_new = R @ pt + t
        transformed_dict[key] = list(pt_new.flatten())
    return transformed_dict

if __name__ == "__main__":
    try:
        # 1. 載入標定參數
        left_data = np.load('./20251125_exp1_雙目參數檔/exp1/20251125_left_single_1_2.npz')
        right_data = np.load('./20251125_exp1_雙目參數檔/exp1/20251125_right_single_1_2.npz')
        stereo_data = np.load('./20251125_exp1_雙目參數檔/exp1/stereo_rt_result_1_2.npz')
        
        mtxL, distL = left_data['camera_matrix'], left_data['dist_coeffs']
        mtxR, distR = right_data['camera_matrix'], right_data['dist_coeffs']
        R_stereo, T_stereo = stereo_data['R'], stereo_data['T']

        # REAL_BASELINE_METERS = 5.913495542522613
        
        reconstructor = StereoReconstructor(mtxL, distL, mtxR, distR, R_stereo, T_stereo, real_baseline=None)
        
        # 2. 載入並計算 3D 點
        with open('correspond_points_15_originimg_2.json', 'r') as f:
            pixel_points = json.load(f)
        with open('vicon_3d_points.json', 'r') as f:
            vicon_points = json.load(f)
            
        stereo_3d_points = {}
        for key, val in pixel_points.items():
            uL, vL = val['left']
            uR, vR = val['right']
            stereo_3d_points[key] = reconstructor.calculate_3d_point(uL, vL, uR, vR)

        # 3. 執行 SVD 座標轉換
        common_keys = sorted([k for k in stereo_3d_points.keys() if k in vicon_points])
        A = np.array([stereo_3d_points[k] for k in common_keys]).T
        B = np.array([vicon_points[k] for k in common_keys]).T
        
        R_opt, t_opt = rigid_transform_3D(A, B)
        transformed_points = apply_transform(stereo_3d_points, R_opt, t_opt)
        
        # 4. 輸出結果與誤差
        errors = [np.linalg.norm(np.array(transformed_points[k]) - np.array(vicon_points[k])) for k in common_keys]
        
        print("轉換矩陣 R:\n", R_opt)
        print("轉換向量 t:\n", t_opt)
        print(f"平均誤差 (RMSE): {np.mean(errors):.4f} m")
        
        with open('./origin_img/final_transformed_points_2.json', 'w') as f:
            json.dump(transformed_points, f, indent=4)
            
    except Exception as e:
        print(f"執行錯誤: {e}")