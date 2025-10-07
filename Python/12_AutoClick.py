import time
import threading
import pyautogui

def autoclick(interval=0.1):
    try:
        while running[0]:
            pyautogui.click()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

def start_autoclicker():
    print("Press Enter to stop autoclicking.")
    thread = threading.Thread(target=autoclick)
    thread.start()
    input()
    running[0] = False
    thread.join()
    print("Autoclicker stopped.")

if __name__ == "__main__":
    running = [True]
    start_autoclicker()