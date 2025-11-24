from djitellopy import Tello
import cv2
import time

def test_color_formats():
    """
    測試不同的顏色格式轉換，找出正確的顏色顯示方式
    按下數字鍵 1-4 來切換不同的顏色格式
    """

    tello = Tello()

    try:
        print("正在連接 Tello 無人機...")
        tello.connect()
        print(f"電池電量: {tello.get_battery()}%")

        print("開啟視訊串流...")
        tello.streamon()
        frame_read = tello.get_frame_read()

        color_mode = 0  # 預設模式
        modes = [
            ("原始格式 (No conversion)", None),
            ("RGB to BGR", cv2.COLOR_RGB2BGR),
            ("BGR to RGB", cv2.COLOR_BGR2RGB),
            ("YUV to BGR", cv2.COLOR_YUV2BGR)
        ]

        print("\n測試顏色格式：")
        print("按 1-4 切換不同的顏色格式")
        print("按 ESC 退出\n")

        while True:
            # 獲取影像
            img = frame_read.frame

            # 根據選擇的模式轉換顏色
            if modes[color_mode][1] is not None:
                img = cv2.cvtColor(img, modes[color_mode][1])

            # 顯示當前模式
            mode_text = f"Mode {color_mode + 1}: {modes[color_mode][0]}"
            cv2.putText(img, mode_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Battery: {tello.get_battery()}%",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Press 1-4 to switch color modes",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # 顯示影像
            cv2.imshow("Color Format Test", img)

            # 處理鍵盤輸入
            key = cv2.waitKey(1) & 0xff

            if key == 27:  # ESC
                break
            elif ord('1') <= key <= ord('4'):
                color_mode = key - ord('1')
                print(f"切換到模式 {color_mode + 1}: {modes[color_mode][0]}")

    except Exception as e:
        print(f"錯誤: {e}")

    finally:
        print("\n正在清理...")
        try:
            tello.streamoff()
        except:
            pass
        cv2.destroyAllWindows()
        try:
            tello.end()
        except:
            pass
        print("測試結束")

if __name__ == "__main__":
    test_color_formats()