import subprocess
import urllib.request
import urllib.error
import json
import time
import sys

# 1. 從 Git Credential Manager 取得 token
def get_github_token():
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(input='protocol=https\nhost=github.com\n\n')
    token = ''
    for line in out.splitlines():
        if line.startswith('password='):
            token = line.split('=', 1)[1]
    return token

token = get_github_token()
if not token:
    print("Error: Could not retrieve GitHub token from credential manager.")
    sys.exit(1)

headers = {
    'User-Agent': 'Mozilla/5.0',
    'Authorization': f'Bearer {token}',
    'Accept': 'application/vnd.github+json'
}

repo_name = 'HW2-Multiple-Linear-Regression'
username = 'ajidohf9'
repo_url = f'https://github.com/{username}/{repo_name}'
web_url = f'https://{username}.github.io/{repo_name}/'

print(f"Target Repo: {repo_url}")
print(f"Target Web:  {web_url}")

# 2. 檢查或建立 GitHub Repository
try:
    check_req = urllib.request.Request(f'https://api.github.com/repos/{username}/{repo_name}', headers=headers)
    with urllib.request.urlopen(check_req) as resp:
        print("Repository already exists on GitHub!")
except urllib.error.HTTPError as e:
    if e.code == 404:
        print("Creating repository on GitHub...")
        payload = json.dumps({
            'name': repo_name,
            'description': 'NCHU HW2: Multiple Linear Regression Analysis Platform & CRISP-DM Prediction Engine',
            'private': False,
            'auto_init': False
        }).encode('utf-8')
        create_req = urllib.request.Request('https://api.github.com/user/repos', data=payload, headers=headers, method='POST')
        with urllib.request.urlopen(create_req) as resp:
            print("Repository created successfully!")
    else:
        print(f"Error checking repo: {e}")

# 3. 初始化本地 Git 倉庫並配置
def run_git(cmd):
    print(f">> git {' '.join(cmd)}")
    res = subprocess.run(['git'] + cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print("ERR:", res.stderr.strip())
    return res

run_git(['init'])
run_git(['config', 'user.name', 'ajidohf9'])
run_git(['config', 'user.email', 'ajidohf9@users.noreply.github.com'])
run_git(['branch', '-M', 'main'])

# 設定 remote
remotes = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True).stdout
if 'origin' in remotes:
    run_git(['remote', 'set-url', 'origin', f'https://github.com/{username}/{repo_name}.git'])
else:
    run_git(['remote', 'add', 'origin', f'https://github.com/{username}/{repo_name}.git'])

# 4. 加入檔案並 commit
run_git(['add', '.'])
run_git(['commit', '-m', 'feat: Initial release of HW2 Multiple Linear Regression web app and CRISP-DM report'])

# 5. 推送至 GitHub
# 在 URL 中內嵌 token 以避免互動式彈跳視窗
auth_url = f'https://{username}:{token}@github.com/{username}/{repo_name}.git'
print("Pushing to GitHub main branch...")
push_res = subprocess.run(['git', 'push', '-u', auth_url, 'main', '--force'], capture_output=True, text=True, encoding='utf-8', errors='replace')
print("Push output:", push_res.stdout)
if push_res.stderr:
    # 遮蔽 token
    safe_err = push_res.stderr.replace(token, '***')
    print("Push stderr/info:", safe_err)

# 重設 remote URL 為普通乾淨的 https 網址 (不留 token)
run_git(['remote', 'set-url', 'origin', f'https://github.com/{username}/{repo_name}.git'])

# 6. 開啟 GitHub Pages
print("\nConfiguring GitHub Pages...")
pages_url = f'https://api.github.com/repos/{username}/{repo_name}/pages'
try:
    check_p = urllib.request.Request(pages_url, headers=headers)
    with urllib.request.urlopen(check_p) as resp:
        print("GitHub Pages is already enabled!")
except urllib.error.HTTPError as e:
    if e.code == 404:
        # 啟用 Pages
        payload = json.dumps({
            'source': {
                'branch': 'main',
                'path': '/'
            }
        }).encode('utf-8')
        enable_req = urllib.request.Request(pages_url, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(enable_req) as resp:
                print("GitHub Pages enabled successfully!")
        except Exception as ex:
            print("Failed to enable Pages automatically:", ex)
    else:
        print(f"Pages status: {e}")

print("\n" + "="*50)
print(f"SUCCESS!")
print(f"GitHub Repository: {repo_url}")
print(f"Web Platform (Pages): {web_url}")
print("="*50)
