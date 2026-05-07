import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAT_FILE = os.path.join(SCRIPT_DIR, "file_monitor.bat")
STARTUP_FOLDER = os.path.join(os.environ['APPDATA'], r"Microsoft\Windows\Start Menu\Programs\Startup")
SHORTCUT_NAME = "文件监控AI整理.lnk"

def install_autostart():
    print("安装自启动...\n")

    if not os.path.exists(BAT_FILE):
        print(f"[错误] 找不到 {BAT_FILE}")
        return False

    try:
        ps_script = f'''
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("{os.path.join(STARTUP_FOLDER, SHORTCUT_NAME)}")
        $Shortcut.TargetPath = "cmd.exe"
        $Shortcut.Arguments = "/c {BAT_FILE}"
        $Shortcut.WorkingDirectory = "{SCRIPT_DIR}"
        $Shortcut.Description = "文件监控AI整理"
        $Shortcut.Save()
        '''
        result = os.system(f'powershell -Command "{ps_script}"')
        if result == 0:
            print(f"[成功] 已添加到启动文件夹")
            print(f"  快捷方式: {os.path.join(STARTUP_FOLDER, SHORTCUT_NAME)}")
            return True
        else:
            print(f"[失败] 添加启动项失败")
            return False
    except Exception as e:
        print(f"[错误] {e}")
        return False

def uninstall_autostart():
    print("卸载自启动...\n")

    shortcut_path = os.path.join(STARTUP_FOLDER, SHORTCUT_NAME)
    if os.path.exists(shortcut_path):
        os.remove(shortcut_path)
        print(f"[成功] 已从启动文件夹移除")
    else:
        print(f"[提示] 启动文件夹中没有找到快捷方式")

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("自启动设置工具")
    print("=" * 50)
    print()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("1. 安装自启动")
        print("2. 卸载自启动")
        print("3. 退出")
        print()
        choice = input("请选择 (1/2/3): ").strip()

    if choice == "1":
        install_autostart()
    elif choice == "2":
        uninstall_autostart()
    elif choice == "3":
        print("退出")
    else:
        print("无效选择")