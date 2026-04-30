import os
import time
import json
import subprocess
from datetime import datetime
from openai import OpenAI

BASE_DIR = r"D:\KNOWLEGE\knowledge"
LINK_DIR = os.path.join(BASE_DIR, "LINK")
NOTE_DIR = os.path.join(BASE_DIR, "NOTE")
STATE_FILE = os.path.join(BASE_DIR, "file_monitor_state.json")
CONFIG_FILE = os.path.join(BASE_DIR, ".unignore")

ARK_API_KEY = "sk-cp-SZKQUWlP9Zceww2G67b1xa7y9djCykErpd9wTl1Q38kNv13wZlTRc4U_qGoiNKE-fmlZK9-oZqTkHk48eL3S2tuXkifrxSiC5ni3DEvS7l7rZWBllY3AkCo"
ARK_BASE_URL = "https://api.minimax.chat/v1"
ARK_MODEL = "MiniMax-M2.7"

def load_extensions():
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_lnk_targets():
    targets = {}
    if not os.path.exists(LINK_DIR):
        return targets

    for item in os.listdir(LINK_DIR):
        if item.endswith('.lnk'):
            lnk_path = os.path.join(LINK_DIR, item)
            target = resolve_lnk_target(lnk_path)
            if target:
                shortcut_name = item.replace('.lnk', '').strip()
                targets[lnk_path] = {'target': target, 'name': shortcut_name}
    return targets

def resolve_lnk_target(lnk_path):
    try:
        ps_script = f"""$shell = New-Object -ComObject WScript.Shell; $shortcut = $shell.CreateShortcut('{lnk_path}'); Write-Output $shortcut.TargetPath"""
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        target = result.stdout.strip()
        return target if target else None
    except Exception:
        return None

def get_all_monitored_files(target_dirs, extensions):
    files = {}
    for lnk_path, info in target_dirs.items():
        base_path = info['target']
        shortcut_name = info['name']
        if not base_path:
            continue
        print(f"  扫描 [{shortcut_name}]: {base_path}")
        try:
            all_files = []
            for ext in extensions:
                ext_clean = ext.strip('*.').lower()
                ps_script = f"""Get-ChildItem '{base_path}' -Recurse -File -Filter '*.{ext_clean}' | Select-Object FullName, LastWriteTime, Length | ConvertTo-Json -Compress"""
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.stdout.strip():
                    try:
                        items = json.loads(result.stdout)
                        if isinstance(items, dict):
                            items = [items]
                        all_files.extend(items)
                    except json.JSONDecodeError:
                        pass
            print(f"    找到 {len(all_files)} 个文件")
            for item in all_files:
                filepath = item['FullName']
                files[filepath] = {
                    'mtime': item['LastWriteTime'],
                    'size': item['Length'],
                    'rel_path': os.path.relpath(filepath, base_path),
                    'shortcut_name': shortcut_name
                }
        except Exception as e:
            print(f"    扫描失败: {e}")
    print(f"  共扫描到 {len(files)} 个文件")
    return files

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def parse_file_with_ai(filepath, content):
    client = OpenAI(
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL
    )

    prompt = f"""请分析以下文件内容，并生成一份简洁的 Markdown 格式笔记。

要求：
1. 提取关键信息和知识点
2. 使用清晰的 Markdown 格式（标题、列表、代码块等）
3. 保留重要的技术细节
4. 文件名：{os.path.basename(filepath)}

文件内容：
{content[:4000]}
"""

    try:
        response = client.chat.completions.create(
            model=ARK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的知识整理助手，擅长提取文档中的关键信息并整理成结构化的 Markdown 笔记。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [AI解析失败] {e}")
        return None

def save_markdown(source_filepath, content, raw_content, shortcut_name):
    rel_path = os.path.relpath(source_filepath, target_base)
    md_filename = os.path.splitext(rel_path)[0] + '.md'
    md_path = os.path.join(NOTE_DIR, shortcut_name, md_filename)

    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {os.path.basename(source_filepath)}\n\n")
        f.write(f"**源文件**: `{source_filepath}`\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("## 原始内容\n\n")
        f.write("```\n")
        f.write(raw_content[:10000])
        if len(raw_content) > 10000:
            f.write(f"\n... (内容过长，已截断前10000字符)")
        f.write("\n```\n\n")
        f.write("---\n\n")
        f.write("## AI 整理笔记\n\n")
        f.write(content)

    return md_path

def read_file_content(filepath, ext):
    if ext in ['.txt', '.py', '.tpl']:
        try:
            ps_script = f"""Get-Content '{filepath}' -Raw -Encoding UTF8"""
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout if result.stdout else None
        except Exception:
            return None
    elif ext == '.docx':
        try:
            from docx import Document
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, 'temp_doc.docx')
            subprocess.run(['powershell', '-Command', f"Copy-Item '{filepath}' '{temp_file}' -Force"], check=True)
            doc = Document(temp_file)
            os.remove(temp_file)
            return "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            print(f"    [跳过] 需要 python-docx 库")
            return None
        except Exception:
            return None
    elif ext == '.xls':
        try:
            import xlrd
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, 'temp_xls.xls')
            subprocess.run(['powershell', '-Command', f"Copy-Item '{filepath}' '{temp_file}' -Force"], check=True)
            wb = xlrd.open_workbook(temp_file)
            os.remove(temp_file)
            sheets = []
            for sheet in wb.sheets():
                rows = []
                for i in range(sheet.nrows):
                    rows.append(" | ".join(str(c.value) for c in sheet.row(i)))
                sheets.append(f"## {sheet.name}\n" + "\n".join(rows))
            return "\n\n".join(sheets)
        except ImportError:
            print(f"    [跳过] 需要 xlrd 库")
            return None
        except Exception:
            return None
    return None

def process_file(filepath, info, target_dirs):
    for lnk_path, lnk_info in target_dirs.items():
        if filepath.startswith(lnk_info['target']):
            print(f"    处理: {filepath}")

            ext = os.path.splitext(filepath)[1].lower()
            raw_content = read_file_content(filepath, ext)

            if not raw_content:
                print(f"    [读取失败或空文件]")
                return None

            ai_content = parse_file_with_ai(filepath, raw_content)
            if ai_content:
                global target_base
                target_base = lnk_info['target']
                md_path = save_markdown(filepath, ai_content, raw_content, lnk_info['name'])
                print(f"    -> {md_path}")
                return md_path
            return None

    return None

def cleanup():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"已删除状态文件: {STATE_FILE}")
    if os.path.exists(NOTE_DIR):
        import shutil
        shutil.rmtree(NOTE_DIR)
        print(f"已删除NOTE目录: {NOTE_DIR}")

def monitor_loop(interval=10, clean_start=False):
    global target_base
    target_base = ""

    if clean_start:
        print("清理旧数据...\n")
        cleanup()
        print()

    extensions = load_extensions()
    print(f"监控文件类型: {extensions}\n")

    target_dirs = get_lnk_targets()
    if not target_dirs:
        print("[错误] 未找到有效的快捷方式目标")
        return

    print("快捷方式目标:")
    for lnk, info in target_dirs.items():
        print(f"  [{info['name']}] -> {info['target']}\n")

    state = load_state()
    print(f"已加载 {len(state)} 个文件的历史状态\n")

    print(f"开始监控 (每 {interval} 秒检查一次)...")
    print("按 Ctrl+C 停止\n")

    while True:
        try:
            current_files = get_all_monitored_files(target_dirs, extensions)

            print(f"\n本次扫描到 {len(current_files)} 个文件")
            print(f"状态文件中 {len(state)} 个文件")

            for filepath, info in current_files.items():
                is_new = filepath not in state
                is_modified = not is_new and info['mtime'] > state[filepath]['mtime']
                is_size_changed = not is_new and info['size'] != state[filepath]['size']

                if is_new:
                    print(f"[新增] {filepath}")
                elif is_modified:
                    print(f"[修改] {filepath}")
                elif is_size_changed:
                    print(f"[大小变化] {filepath}")
                else:
                    print(f"[跳过(未变化)] {os.path.basename(filepath)}")

                if is_new or is_modified or is_size_changed:
                    result = process_file(filepath, info, target_dirs)
                    if result:
                        print(f"    [成功] -> {result}")
                    else:
                        print(f"    [失败]")

                state[filepath] = info
                save_state(state)

            for filepath in list(state.keys()):
                if filepath not in current_files:
                    print(f"[删除] {filepath}")
                    del state[filepath]
                    save_state(state)

        except Exception as e:
            print(f"[监控出错] {e}")

        time.sleep(interval)

if __name__ == "__main__":
    print("=" * 60)
    print("文件监控 AI 整理系统")
    print("=" * 60)
    print()
    monitor_loop(interval=10, clean_start=False)