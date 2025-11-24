from djitellopy import Tello
import cv2
import time

def main():
    """
    主程式：控制 Tello 無人機並顯示即時影像

    鍵盤控制：
    - W: 向前移動
    - S: 向後移動
    - A: 向左移動
    - D: 向右移動
    - Q: 逆時針旋轉
    - E: 順時針旋轉
    - 空白鍵: 向上移動
    - X: 向下移動
    - ESC: 結束程式並降落
    """

    # 初始化 Tello 無人機
    tello = Tello()

    try:
        # 連接無人機
        print("正在連接 Tello 無人機...")
        tello.connect()
        print(f"電池電量: {tello.get_battery()}%")

        # 開啟視訊串流
        print("開啟視訊串流...")
        tello.streamon()
        frame_read = tello.get_frame_read()

        # 起飛
        print("無人機起飛...")
        tello.takeoff()
        time.sleep(2)  # 等待穩定

        print("\n控制說明:")
        print("W/S: 前進/後退")
        print("A/D: 左移/右移")
        print("Q/E: 逆時針/順時針旋轉")
        print("空白鍵: 上升")
        print("X: 下降")
        print("ESC: 結束並降落\n")

        # 主控制迴圈
        while True:
            # 獲取並顯示影像
            img = frame_read.frame

            # 轉換顏色格式從 BGR 到 RGB（如果需要的話）
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 若顏色仍有問題，可嘗試以下轉換
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 在影像上顯示控制提示
            cv2.putText(img, "Battery: {}%".format(tello.get_battery()),
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, "Press ESC to land and exit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # 顯示影像
            cv2.imshow("Tello Drone Control", img)

            # 處理鍵盤輸入
            key = cv2.waitKey(1) & 0xff

            if key == 27:  # ESC 鍵
                print("收到退出指令，準備降落...")
                break
            elif key == ord('w'):
                print("向前移動 30cm")
                tello.move_forward(30)
            elif key == ord('s'):
                print("向後移動 30cm")
                tello.move_back(30)
            elif key == ord('a'):
                print("向左移動 30cm")
                tello.move_left(30)
            elif key == ord('d'):
                print("向右移動 30cm")
                tello.move_right(30)
            elif key == ord('e'):
                print("順時針旋轉 30度")
                tello.rotate_clockwise(30)
            elif key == ord('q'):
                print("逆時針旋轉 30度")
                tello.rotate_counter_clockwise(30)
            elif key == 32:  # 空白鍵
                print("向上移動 30cm")
                tello.move_up(30)
            elif key == ord('x'):
                print("向下移動 30cm")
                tello.move_down(30)

    except Exception as e:
        print(f"發生錯誤: {e}")

    finally:
        # 確保無人機安全降落並關閉連接
        print("正在降落...")
        try:
            tello.land()
            time.sleep(2)
        except:
            pass

        print("關閉視訊串流...")
        try:
            tello.streamoff()
        except:
            pass

        print("關閉視窗...")
        cv2.destroyAllWindows()

        print("結束連接...")
        try:
            tello.end()
        except:
            pass

        print("程式結束")

if __name__ == "__main__":
    main()