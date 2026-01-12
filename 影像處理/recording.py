import cv2

def record_synchronous_videos(rtsp_url_1, rtsp_url_2, output_file_1='output1.avi', output_file_2='output2.avi'):
    cap1 = cv2.VideoCapture(rtsp_url_1)
    cap2 = cv2.VideoCapture(rtsp_url_2)

    if not cap1.isOpened():
        print("無法開啟第一台攝影機")
        return
    if not cap2.isOpened():
        print("無法開啟第二台攝影機")
        return

    # 取得影像的寬度和高度
    frame_width_1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定義編碼器和建立 VideoWriter 物件
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_file_1, fourcc, 20.0, (frame_width_1, frame_height_1))
    out2 = cv2.VideoWriter(output_file_2, fourcc, 20.0, (frame_width_2, frame_height_2))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1:
            print("無法讀取第一台攝影機的影像")
            break
        if not ret2:
            print("無法讀取第二台攝影機的影像")
            break

        # 寫入影像到檔案
        out1.write(frame1)
        out2.write(frame2)

        # 顯示兩台攝影機的畫面
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)

        # 按下 q 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

# 使用範例
rtsp_url_1 = "rtsp://ai4dt:ai4dt@192.168.0.212:8543/ts"
rtsp_url_2 = "rtsp://ai4dt:ai4dt@192.168.0.212:8544/ts"
record_synchronous_videos(rtsp_url_1, rtsp_url_2)