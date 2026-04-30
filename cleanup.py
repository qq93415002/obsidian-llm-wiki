import os
import shutil

BASE_DIR = r"D:\KNOWLEGE\knowledge"
STATE_FILE = os.path.join(BASE_DIR, "file_monitor_state.json")
NOTE_DIR = os.path.join(BASE_DIR, "NOTE")

def cleanup():
    print("开始清理...\n")

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"已删除状态文件: {STATE_FILE}")
    else:
        print(f"状态文件不存在: {STATE_FILE}")

    if os.path.exists(NOTE_DIR):
        shutil.rmtree(NOTE_DIR)
        print(f"已删除NOTE目录: {NOTE_DIR}")
    else:
        print(f"NOTE目录不存在: {NOTE_DIR}")

    print("\n清理完成!")

if __name__ == "__main__":
    cleanup()