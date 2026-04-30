import re

with open(r'D:\KNOWLEGE\wiki\wiki.toml', 'r') as f:
    content = f.read()

content = re.sub(r'fast = ".*"', 'fast = "MiniMax-M2.7"', content)
content = re.sub(r'heavy = ".*"', 'heavy = "MiniMax-M2.7"', content)
content = re.sub(r'url = ".*"', 'url = "https://api.minimax.chat/v1"', content)
content = re.sub(r'timeout = \d+', 'timeout = 120', content)

with open(r'D:\KNOWLEGE\wiki\wiki.toml', 'w') as f:
    f.write(content)

print('Done')
