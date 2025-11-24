from djitellopy import Tello
import cv2
import time
import threading
from datetime import datetime

class TelloDroneController:
    """
    進階 Tello 無人機控制器
    提供即時影像串流、鍵盤控制、狀態監控等功能
    """

    def __init__(self):
        self.tello = Tello()
        self.frame = None
        self.frame_read = None
        self.is_flying = False
        self.recording = False
        self.video_writer = None
        self.movement_distance = 30  # 預設移動距離 (cm)
        self.rotation_angle = 30      # 預設旋轉角度 (度)

    def connect_drone(self):
        """連接無人機並初始化"""
        try:
            print("=" * 50)
            print("正在連接 Tello 無人機...")
            self.tello.connect()
            print(f"✓ 連接成功")
            print(f"電池電量: {self.tello.get_battery()}%")
            print(f"溫度: {self.tello.get_temperature()}°C")
            print(f"飛行時間: {self.tello.get_flight_time()}秒")
            print("=" * 50)
            return True
        except Exception as e:
            print(f"✗ 連接失敗: {e}")
            return False

    def start_video_stream(self):
        """開啟視訊串流"""
        try:
            print("開啟視訊串流...")
            self.tello.streamon()
            self.frame_read = self.tello.get_frame_read()
            print("✓ 視訊串流已開啟")
            return True
        except Exception as e:
            print(f"✗ 視訊串流開啟失敗: {e}")
            return False

    def takeoff_safe(self):
        """安全起飛"""
        if not self.is_flying:
            battery = self.tello.get_battery()
            if battery < 20:
                print(f"⚠️  電量過低 ({battery}%)，無法起飛")
                return False

            print("無人機起飛中...")
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(2)  # 等待穩定
            print("✓ 起飛成功")
            return True
        else:
            print("無人機已在飛行中")
            return False

    def land_safe(self):
        """安全降落"""
        if self.is_flying:
            print("無人機降落中...")
            self.tello.land()
            self.is_flying = False
            time.sleep(2)
            print("✓ 降落成功")
            return True
        else:
            print("無人機未在飛行中")
            return False

    def emergency_stop(self):
        """緊急停止"""
        print("⚠️  緊急停止！")
        try:
            self.tello.emergency()
            self.is_flying = False
        except:
            pass

    def start_recording(self):
        """開始錄影"""
        if not self.recording and self.frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tello_recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0,
                                               (self.frame.shape[1], self.frame.shape[0]))
            self.recording = True
            print(f"🔴 開始錄影: {filename}")
            return filename
        return None

    def stop_recording(self):
        """停止錄影"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.recording = False
            print("⬜ 錄影已停止")

    def take_photo(self):
        """拍照"""
        if self.frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tello_photo_{timestamp}.jpg"
            cv2.imwrite(filename, self.frame)
            print(f"📸 照片已儲存: {filename}")
            return filename
        return None

    def draw_hud(self, img):
        """繪製 HUD (平視顯示器) 資訊"""
        height, width = img.shape[:2]

        # 電池資訊
        battery = self.tello.get_battery()
        battery_color = (0, 255, 0) if battery > 50 else (0, 165, 255) if battery > 20 else (0, 0, 255)
        cv2.putText(img, f"Battery: {battery}%",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, battery_color, 2)

        # 高度資訊
        try:
            height_info = self.tello.get_height()
            cv2.putText(img, f"Height: {height_info}cm",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass

        # 飛行狀態
        status = "Flying" if self.is_flying else "Landed"
        status_color = (0, 255, 0) if self.is_flying else (128, 128, 128)
        cv2.putText(img, f"Status: {status}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 錄影狀態
        if self.recording:
            cv2.circle(img, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(img, "REC", (width - 80, 37),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 控制提示
        controls = [
            "Controls:",
            "W/S: Forward/Back",
            "A/D: Left/Right",
            "Q/E: Rotate L/R",
            "Space: Up",
            "X: Down",
            "T: Takeoff",
            "L: Land",
            "P: Photo",
            "V: Video Toggle",
            "G: Emergency",
            "ESC: Exit"
        ]

        y_pos = height - 250
        for control in controls:
            cv2.putText(img, control, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += 20

        return img

    def process_keyboard_input(self, key):
        """處理鍵盤輸入"""
        if key == 27:  # ESC
            return False  # 結束程式

        elif key == ord('t') or key == ord('T'):
            self.takeoff_safe()

        elif key == ord('l') or key == ord('L'):
            self.land_safe()

        elif key == ord('g') or key == ord('G'):  # G 鍵 - 緊急停止
            self.emergency_stop()

        elif key == ord('p') or key == ord('P'):
            self.take_photo()

        elif key == ord('v') or key == ord('V'):
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()

        # 移動控制 (只有在飛行中才能移動)
        elif self.is_flying:
            if key == ord('w'):
                print(f"↑ 向前移動 {self.movement_distance}cm")
                self.tello.move_forward(self.movement_distance)
            elif key == ord('s'):
                print(f"↓ 向後移動 {self.movement_distance}cm")
                self.tello.move_back(self.movement_distance)
            elif key == ord('a'):
                print(f"← 向左移動 {self.movement_distance}cm")
                self.tello.move_left(self.movement_distance)
            elif key == ord('d'):
                print(f"→ 向右移動 {self.movement_distance}cm")
                self.tello.move_right(self.movement_distance)
            elif key == ord('q'):
                print(f"↺ 逆時針旋轉 {self.rotation_angle}度")
                self.tello.rotate_counter_clockwise(self.rotation_angle)
            elif key == ord('e'):
                print(f"↻ 順時針旋轉 {self.rotation_angle}度")
                self.tello.rotate_clockwise(self.rotation_angle)
            elif key == 32:  # 空白鍵 - 向上移動
                print(f"⬆ 向上移動 {self.movement_distance}cm")
                self.tello.move_up(self.movement_distance)
            elif key == ord('x'):  # X 鍵 - 向下移動
                print(f"⬇ 向下移動 {self.movement_distance}cm")
                self.tello.move_down(self.movement_distance)

            # 翻轉動作
            elif key == ord('1'):
                print("🔄 向前翻轉")
                self.tello.flip_forward()
            elif key == ord('2'):
                print("🔄 向後翻轉")
                self.tello.flip_back()
            elif key == ord('3'):
                print("🔄 向左翻轉")
                self.tello.flip_left()
            elif key == ord('4'):
                print("🔄 向右翻轉")
                self.tello.flip_right()

        return True  # 繼續執行

    def run(self):
        """主程式運行"""
        if not self.connect_drone():
            return

        if not self.start_video_stream():
            return

        print("\n" + "=" * 50)
        print("Tello 無人機控制系統已啟動")
        print("=" * 50 + "\n")

        try:
            while True:
                # 獲取影像
                self.frame = self.frame_read.frame

                # 轉換顏色格式從 RGB 到 BGR（修正顏色問題）
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

                # 繪製 HUD
                display_frame = self.draw_hud(self.frame.copy())

                # 顯示影像
                cv2.imshow("Tello Drone Controller", display_frame)

                # 如果正在錄影，寫入影片
                if self.recording and self.video_writer:
                    self.video_writer.write(self.frame)

                # 處理鍵盤輸入
                key = cv2.waitKey(1) & 0xff
                if key != 255:  # 有按鍵輸入
                    if not self.process_keyboard_input(key):
                        break

        except KeyboardInterrupt:
            print("\n收到中斷信號")

        except Exception as e:
            print(f"發生錯誤: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """清理資源"""
        print("\n正在清理資源...")

        # 停止錄影
        if self.recording:
            self.stop_recording()

        # 安全降落
        if self.is_flying:
            self.land_safe()

        # 關閉視訊串流
        try:
            self.tello.streamoff()
            print("✓ 視訊串流已關閉")
        except:
            pass

        # 關閉視窗
        cv2.destroyAllWindows()

        # 結束連接
        try:
            self.tello.end()
            print("✓ 連接已結束")
        except:
            pass

        print("程式結束")

def main():
    controller = TelloDroneController()
    controller.run()

if __name__ == "__main__":
    main()