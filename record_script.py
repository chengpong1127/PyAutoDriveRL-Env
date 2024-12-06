import cv2
import numpy as np
import subprocess
import time
import mss
import win32con
import win32gui

# 超參數設定區
END_TIME = 30  # 結束時間，單位秒
VIDEO_PATH = "output_video.mp4"  # 儲存影片的路徑
VIDEO_WIDTH = 720  # 視訊寬度
VIDEO_HEIGHT = 640  # 視訊高度
FPS = 30  # 每秒幾幀

# 設定視頻錄製編碼
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))


# 倒數計時顯示
def countdown_display(frame, remaining_time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # 白色
    thickness = 30
    font_scale = 20

    # 将剩余时间转换为字符串
    text = str(remaining_time)

    # 获取文本的宽度和高度
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # 计算文本的左上角坐标，使其居中
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (frame.shape[0] + text_height) // 2

    # 在画面上绘制倒计时文本
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def text_log_display(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # 白色
    thickness = 2
    font_scale = 1
    # 顯示text在畫面上方
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    text_x = (frame.shape[1] - text_width) // 2
    text_y = 75
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def get_all_hwnd(hwnd, mouse):
    if (win32gui.IsWindow(hwnd) and
            win32gui.IsWindowEnabled(hwnd) and
            win32gui.IsWindowVisible(hwnd)):
        hwnd_map.update({hwnd: win32gui.GetWindowText(hwnd)})


hwnd_map = {}
win32gui.EnumWindows(get_all_hwnd, 0)


def put_video_foreground():
    for h, t in hwnd_map.items():
        if t:
            if t == 'Car':
                # h 为想要放到最前面的窗口句柄
                print(h)

                # 被其他窗口遮挡，调用后放到最前面
                win32gui.SetForegroundWindow(h)

                # 解决被最小化的情况
                win32gui.ShowWindow(h, win32con.SW_RESTORE)


def record_video():
    hwnd = win32gui.FindWindow(None, 'Car')  # 替换 'Car' 为你的窗口标题
    if not hwnd:
        print("Error: 窗口未找到")
        return

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    put_video_foreground()
    subprocess.Popen([r"C:\Users\Bacon\anaconda3\envs\torch\python.exe", "ResetScript.py"])

    # 使用 mss 捕获窗口
    with mss.mss() as sct:
        monitor = {"top": top, "left": left + 10, "width": right - left - 20, "height": bottom - top}

        start_time = time.time()

        try:

            # 倒數5秒
            for remaining_time in range(5, 0, -1):
                for _ in range(FPS):
                    img = np.array(sct.grab(monitor))  # 捕获窗口内容
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为 BGR 格式
                    img = countdown_display(img, remaining_time)  # 添加倒数计时
                    out.write(cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT)))  # 写入视频
                    img = text_log_display(img, "Initializing...")

                    cv2.imshow("Recording", img)  # 显示画面
                    cv2.waitKey(1)

            # 開始執行模型推論
            print("Starting inference...")
            proc = subprocess.Popen([r"C:\Users\Bacon\anaconda3\envs\torch\python.exe", "inference_stablebaseline_template.py"])

            # 紀錄影片直到結束時間
            while time.time() - start_time < END_TIME:
                img = np.array(sct.grab(monitor))  # 捕获窗口内容
                img = text_log_display(img, "Running inference_template.py...")
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为 BGR 格式
                out.write(cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT)))  # 写入视频
                cv2.imshow("Recording", img)  # 显示画面

                if cv2.waitKey(1000 // FPS) & 0xFF == 27:  # 按下 ESC 键退出
                    break

            img = np.array(sct.grab(monitor))  # 捕获窗口内容
            img = text_log_display(img, "Recording finished, showing final frame...")
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为 BGR 格式

            # 停止子程序
            proc.terminate()
            proc.wait()  # 等待子程序完全結束

            # 錄製結束，停在最後一幀並顯示3秒
            print("Recording finished, showing final frame...")
            for _ in range(3 * FPS):  # 显示3秒的最后一帧
                out.write(cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT)))
                cv2.imshow("Recording", img)  # 显示画面
                if cv2.waitKey(1000 // FPS) & 0xFF == 27:
                    break

            # 淡出效果
            for i in range(30):  # 淡出30幀
                alpha = 1 - i / 30.0
                faded_frame = cv2.addWeighted(img, alpha, np.zeros_like(img), 0, 0)
                out.write(cv2.resize(faded_frame, (VIDEO_WIDTH, VIDEO_HEIGHT)))
                cv2.imshow("Recording", faded_frame)
                if cv2.waitKey(1000 // FPS) & 0xFF == 27:
                    break

        finally:
            # 確保所有資源都被正確釋放
            print("Releasing resources...")
            out.release()
            cv2.destroyAllWindows()


# 主程式開始
if __name__ == "__main__":
    record_video()
