# Codex API/GPT 双模式切换与历史保留复现手册

更新时间：2026-04-20
适用环境：Linux + VS Code Remote SSH + 本地存在 ~/.codex 目录

本版新增：rightcode 配置修复流程（base_url = https://right.codes/codex/v1）以及 provider 校验命令。

## 1. 目标

在同一台机器上实现以下能力：

- 在 API 模式 与 GPT 账号模式之间稳定切换
- 切到 GPT 模式时不再误走 base_url 网关
- 保留两种模式下的本地聊天历史
- 提供跨模式统一历史查看能力（绕过插件 UI 过滤）

## 2. 关键结论（先看）

- 聊天记录看起来“丢失”，多数情况不是数据丢失，而是插件 UI 会按 provider/认证方式过滤显示。
- 本地历史文件只要合并成功，数据仍在 ~/.codex 下。
- 要避免 GPT 模式仍走 API 网关，必须确保：
  - auth_mode = chatgpt
  - model_provider = openai
  - config.toml 中不再保留 rightcode 的 base_url 生效路径

## 3. 一次性部署脚本

将切换脚本放到 ~/.local/bin/codex-auth-switch，并赋予可执行权限。

脚本支持命令：

- status
- save-api [--force]
- save-gpt [--force]
- set-api-provider [provider_name] <base_url> <api_key>
- save-provider <provider_name>
- use-provider <provider_name>
- list-providers
- api
- gpt
- sync-history
- history [N]

建议直接使用本仓库已有脚本逻辑（本次会话已验证）：

- ~/.local/bin/codex-auth-switch

然后执行：

```bash
chmod +x ~/.local/bin/codex-auth-switch
command -v codex-auth-switch
```

## 4. 首次初始化流程（新机器）

### 步骤 1：保存 API 快照

```bash
codex-auth-switch api
codex-auth-switch save-api
```

### 步骤 2：在 VS Code 插件中登录 GPT 账号

登录后确认 ~/.codex/auth.json 中出现：

- auth_mode: chatgpt
- tokens 非空

### 步骤 3：切到 GPT 并保存 GPT 快照

```bash
codex-auth-switch gpt
codex-auth-switch save-gpt
```

### 步骤 4：合并历史

```bash
codex-auth-switch sync-history
```

### 步骤 5：检查状态

```bash
codex-auth-switch status
```

期望看到：

- profile api: yes
- profile gpt: yes
- history api: yes
- history gpt: yes
- GPT 模式下 model_provider = openai
- GPT 模式下 base_url: <not set>

## 5. 日常使用流程

### 在 API/GPT 间切换

```bash
codex-auth-switch api
# 或
codex-auth-switch gpt
```

### 切换后同步一次历史（推荐）

```bash
codex-auth-switch sync-history
```

### 根据 provider 名称切换不同 API 网关

说明：支持多份 provider 配置并按名称切换，例如 `rightcode`、`ikuncode`、`custom-a`。

```bash
codex-auth-switch set-api-provider ikuncode "https://api.ikuncode.cc/v1" "<ikuncode_api_key>"
codex-auth-switch set-api-provider rightcode "https://right.codes/codex/v1" "<rightcode_api_key>"
```

如果不写 provider_name，会默认保存到 `api`：

```bash
codex-auth-switch set-api-provider "https://example.com/v1" "sk-xxx"
```

切换到已保存 provider：

```bash
codex-auth-switch use-provider ikuncode
```

查看所有已保存 provider：

```bash
codex-auth-switch list-providers
```

执行效果：

- 当前认证切到 `auth_mode=apikey`
- `model_provider` 切到你提供的 provider 名称（例如 `ikuncode`）
- `base_url` 更新为传入的新 URL
- 自动保存为该 provider 的 profile

建议校验（避免 provider 名称与 URL 错配）：

```bash
codex-auth-switch status
grep -A 4 "^\[model_providers.rightcode\]" ~/.codex/config.toml
```

期望看到：

- `model_provider = "rightcode"`
- `[model_providers.rightcode]` 下 `base_url = "https://right.codes/codex/v1"`

### 查看统一历史（跨 provider）

```bash
codex-auth-switch history
codex-auth-switch history 50
```

说明：history 命令是本地统一视图，可看到 rightcode 和 openai 两类会话。即使插件面板不显示，也可在这里确认历史仍在。

补充：`sync-history` 现在会合并所有 provider 的历史快照，不再只限 api/gpt。

## 6. 常见问题与处理

### 问题 A：执行 gpt 后提示 GPT profile missing

现象：

- [WARN] GPT profile missing ...

处理：

1. 先在插件内确认 GPT 登录完成
2. 再执行：

```bash
codex-auth-switch gpt
codex-auth-switch save-gpt
```

### 问题 D：误执行 save-api/save-gpt 覆盖错误 profile

现象：

- 在 gpt 模式执行 save-api，或在 api 模式执行 save-gpt。

当前行为（已加补丁）：

- 默认会拒绝保存并报错，防止覆盖已有 profile。
- 如确实需要跨模式强制保存，必须显式加 `--force`。

示例：

```bash
codex-auth-switch save-api --force
codex-auth-switch save-gpt --force
```

### 问题 E：需要动态替换新的 API 网关地址和 Key

处理：

```bash
codex-auth-switch set-api-provider <provider_name> "<base_url>" "<api_key>"
```

然后检查：

```bash
codex-auth-switch status
grep -A 4 "^\[model_providers.<provider_name>\]" ~/.codex/config.toml
```

说明：`status` 主要用于看当前模式与 provider 名称；provider 的 URL 以 `config.toml` 对应段落为准。

### 问题 F：有多个 provider（例如 3 类）需要长期切换

建议流程：

1. 首次录入每个 provider 一次：

```bash
codex-auth-switch set-api-provider rightcode "<url1>" "<key1>"
codex-auth-switch set-api-provider ikuncode "<url2>" "<key2>"
codex-auth-switch set-api-provider custom "<url3>" "<key3>"
```

2. 日常切换：

```bash
codex-auth-switch use-provider rightcode
codex-auth-switch use-provider ikuncode
codex-auth-switch use-provider custom
```

3. 每次切换后如需确保本地历史并集：

```bash
codex-auth-switch sync-history
codex-auth-switch history 100
```

### 问题 G：rightcode 的配置被误写成其他网关

现象：

- `model_provider = "rightcode"`，但 `base_url` 不是 `https://right.codes/codex/v1`。

修复：

```bash
codex-auth-switch set-api-provider rightcode "https://right.codes/codex/v1" "<rightcode_api_key>"
codex-auth-switch save-provider rightcode
```

校验：

```bash
grep -A 4 "^\[model_providers.rightcode\]" ~/.codex/config.toml
cat ~/.codex/profiles/providers/rightcode/config.toml | grep -A 4 "^\[model_providers.rightcode\]"
```

### 问题 B：GPT 模式仍像在走 API 网关

检查：

```bash
codex-auth-switch status
```

若 GPT 模式下不是 model_provider = openai，重新执行：

```bash
codex-auth-switch gpt
codex-auth-switch save-gpt
```

### 问题 C：GPT 面板看不到 API 历史

结论：

- 常见原因为 UI 过滤，不是数据丢失。

处理：

```bash
codex-auth-switch sync-history
codex-auth-switch history 100
```

如果 history 命令里能同时看到 rightcode/openai，会话已保留。

## 7. 复现验收清单

在另一台机器完成以下检查即算复现成功：

1. codex-auth-switch status 在 api 和 gpt 两种模式都能正确变化
2. gpt 模式 status 显示 model_provider = openai
3. 执行 sync-history 后，history 命令可列出跨 provider 会话
4. profiles 与 history 快照文件已建立：
   - ~/.codex/profiles/auth.api.json
   - ~/.codex/profiles/auth.gpt.json
   - ~/.codex/profiles/history/api/session_index.jsonl
   - ~/.codex/profiles/history/gpt/session_index.jsonl
5. provider 快照目录中包含业务网关配置（例如 rightcode/ikuncode）：
  - ~/.codex/profiles/providers/rightcode/auth.json
  - ~/.codex/profiles/providers/rightcode/config.toml
  - ~/.codex/profiles/providers/ikuncode/auth.json
  - ~/.codex/profiles/providers/ikuncode/config.toml

## 8. 安全建议

- 不要把 ~/.codex/auth.json、token、api key 提交到仓库。
- 建议保留文件权限：

```bash
chmod 600 ~/.codex/auth.json ~/.codex/config.toml
```

- 若 token 或 key 疑似泄漏，立即轮换。

## 9. 本次会话沉淀内容

本次沉淀的能力点：

- 修复切换到 GPT 后仍走 API 网关的问题
- 自动化创建 GPT 快照（已登录情况下）
- 本地历史并集合并（索引 + sessions）
- 跨模式历史查看命令（history [N]）

以上流程可直接迁移到其他机器执行。

## 10. 附录：脚本分发与版本一致性

说明：命令名是 `codex-auth-switch`（不是 `sodex-auth-switch`）。
说明：由于脚本迭代较快，不再在文档内内嵌“固定版本”全量代码，避免与线上使用版本漂移。

### 10.1 从当前机器导出已验证脚本（推荐）

在源机器执行：

```bash
chmod +x ~/.local/bin/codex-auth-switch
sha256sum ~/.local/bin/codex-auth-switch
```

复制到目标机器（任选一种）：

```bash
# 方式 A：scp
scp ~/.local/bin/codex-auth-switch <user>@<target-host>:~/.local/bin/codex-auth-switch

# 方式 B：rsync
rsync -av ~/.local/bin/codex-auth-switch <user>@<target-host>:~/.local/bin/
```

目标机器执行：

```bash
chmod +x ~/.local/bin/codex-auth-switch
command -v codex-auth-switch
codex-auth-switch --help
```

### 10.2 目标机器安装后快速验收

```bash
codex-auth-switch status
codex-auth-switch list-providers
codex-auth-switch sync-history
codex-auth-switch history 20
```

验收关注点：

- `status` 中 `model_provider` 与 `base_url` 对应关系正确。
- `list-providers` 能列出已保存的 provider（例如 rightcode、ikuncode）。
- `history` 能看到跨 provider 的本地并集会话。

### 10.3 最小回归测试命令（建议保存）

```bash
set -e
codex-auth-switch use-provider rightcode
codex-auth-switch status
codex-auth-switch use-provider ikuncode
codex-auth-switch status
codex-auth-switch gpt
codex-auth-switch status
codex-auth-switch sync-history
codex-auth-switch history 30
```

### 10.4 当前机器最新脚本（完整）

来源：`~/.local/bin/codex-auth-switch`

```bash
#!/usr/bin/env bash
set -euo pipefail

CODEX_DIR="${HOME}/.codex"
AUTH_FILE="${CODEX_DIR}/auth.json"
CONFIG_FILE="${CODEX_DIR}/config.toml"
PROFILE_DIR="${CODEX_DIR}/profiles"
PROVIDER_PROFILE_DIR="${PROFILE_DIR}/providers"
SESSIONS_DIR="${CODEX_DIR}/sessions"
SESSION_INDEX_FILE="${CODEX_DIR}/session_index.jsonl"
HISTORY_DIR="${PROFILE_DIR}/history"

usage() {
  cat <<'EOF'
Usage:
  codex-auth-switch status
  codex-auth-switch save-api [--force]
  codex-auth-switch save-gpt [--force]
  codex-auth-switch set-api-provider [provider_name] <base_url> <api_key>
  codex-auth-switch save-provider <provider_name>
  codex-auth-switch use-provider <provider_name>
  codex-auth-switch list-providers
  codex-auth-switch api
  codex-auth-switch gpt
  codex-auth-switch sync-history
  codex-auth-switch history [N]

Commands:
  status    Show current auth mode and whether profile snapshots exist.
  save-api  Save current ~/.codex/{auth.json,config.toml} as API profile snapshot.
            By default, only allowed when current mode is api.
            Use --force to override mode check.
  save-gpt  Save current ~/.codex/{auth.json,config.toml} as GPT profile snapshot.
            By default, only allowed when current mode is gpt.
            Use --force to override mode check.
  set-api-provider Configure API provider base_url/api_key and switch current auth to apikey.
            If provider_name is omitted, defaults to api.
            Saves provider config under named provider profile.
  save-provider Save current ~/.codex/{auth.json,config.toml} as named provider profile.
  use-provider  Switch to named provider profile and merge history snapshots.
  list-providers List all saved provider profiles.
  api       Switch to API profile snapshot (if missing, fallback to set auth_mode=apikey).
  gpt       Switch to GPT profile snapshot (if missing, fallback to set auth_mode=chatgpt).
  sync-history Merge local history indices/files from all snapshots into current ~/.codex.
  history   Show merged local history across providers (default latest 20 rows).

Recommended one-time setup:
  1) You are currently in API mode -> run: codex-auth-switch save-api
  2) In VS Code/Codex, sign in with your GPT account once.
  3) Run: codex-auth-switch save-gpt
  4) Then switch anytime with: codex-auth-switch api / codex-auth-switch gpt
EOF
}

ensure_files() {
  mkdir -p "${PROFILE_DIR}" "${PROVIDER_PROFILE_DIR}" "${HISTORY_DIR}" "${SESSIONS_DIR}"
  if [[ ! -f "${AUTH_FILE}" ]]; then
    echo "[ERROR] Missing ${AUTH_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Missing ${CONFIG_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${SESSION_INDEX_FILE}" ]]; then
    : > "${SESSION_INDEX_FILE}"
  fi
}

current_mode_label() {
  python3 - "$AUTH_FILE" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    mode = (json.load(f).get('auth_mode') or '').strip().lower()
print('gpt' if mode == 'chatgpt' else 'api')
PY
}

capture_history_snapshot() {
  local name="$1"
  local target="${HISTORY_DIR}/${name}"
  mkdir -p "${target}/sessions"
  cp -f "${SESSION_INDEX_FILE}" "${target}/session_index.jsonl" 2>/dev/null || :
  if [[ -d "${SESSIONS_DIR}" ]]; then
    cp -a "${SESSIONS_DIR}/." "${target}/sessions/" 2>/dev/null || :
  fi
}

sync_history_from_snapshots() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap '[[ -n "${tmp_dir:-}" ]] && rm -rf "${tmp_dir}"' RETURN

  local merged_index="${tmp_dir}/merged_session_index.jsonl"
  : > "${merged_index}"

  # Merge all history indices, not only api/gpt.
  [[ -f "${SESSION_INDEX_FILE}" ]] && cat "${SESSION_INDEX_FILE}" >> "${merged_index}"
  for file in "${HISTORY_DIR}"/*/session_index.jsonl; do
    [[ -f "${file}" ]] && cat "${file}" >> "${merged_index}"
  done

  python3 - "${merged_index}" "${SESSION_INDEX_FILE}" <<'PY'
import json
import sys

src = sys.argv[1]
dst = sys.argv[2]
by_id = {}

with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sid = obj.get('id')
        if not sid:
            continue
        prev = by_id.get(sid)
        if not prev or (obj.get('updated_at') or '') > (prev.get('updated_at') or ''):
            by_id[sid] = obj

items = sorted(by_id.values(), key=lambda x: x.get('updated_at') or '', reverse=True)
with open(dst, 'w', encoding='utf-8') as f:
    for obj in items:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
PY

  for src in "${SESSIONS_DIR}" "${HISTORY_DIR}"/*/sessions; do
    if [[ -d "${src}" ]]; then
      cp -a "${src}/." "${SESSIONS_DIR}/" 2>/dev/null || :
    fi
  done

  chmod 600 "${SESSION_INDEX_FILE}" || true
}

show_history() {
  local limit="${1:-20}"
  python3 - "$CODEX_DIR" "$limit" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
limit = int(sys.argv[2]) if str(sys.argv[2]).isdigit() else 20
index_path = base / 'session_index.jsonl'
if not index_path.exists():
    print('no session_index found')
    raise SystemExit(0)

records = []
with index_path.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sid = obj.get('id')
        if not sid:
            continue
        records.append({
            'id': sid,
            'updated_at': obj.get('updated_at', ''),
            'thread_name': obj.get('thread_name', ''),
        })

session_files = list((base / 'sessions').glob('*/*/*/*.jsonl'))
provider_by_id = {}
for fp in session_files:
    try:
        first = fp.open('r', encoding='utf-8').readline().strip()
        if not first:
            continue
        meta = json.loads(first)
        payload = meta.get('payload', {})
        sid = payload.get('id')
        if sid:
            provider_by_id[sid] = payload.get('model_provider', 'unknown')
    except Exception:
        continue

records.sort(key=lambda x: x['updated_at'], reverse=True)
rows = records[:max(limit, 1)]
print('updated_at | provider | id | thread_name')
for r in rows:
    provider = provider_by_id.get(r['id'], 'unknown')
    name = (r['thread_name'] or '').replace('\n', ' ').strip()
    print(f"{r['updated_at']} | {provider} | {r['id']} | {name}")
PY
}

set_auth_mode_fallback() {
  local mode="$1"
  python3 - "$AUTH_FILE" "$mode" <<'PY'
import json, sys
path = sys.argv[1]
mode = sys.argv[2]

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if mode == 'apikey':
    data['auth_mode'] = 'apikey'
elif mode == 'chatgpt':
    data['auth_mode'] = 'chatgpt'
    data.pop('OPENAI_API_KEY', None)
else:
    raise SystemExit('unknown mode')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write('\n')
PY
}

set_api_provider_credentials() {
  local provider_name="$1"
  local base_url="$2"
  local api_key="$3"

  python3 - "$AUTH_FILE" "$api_key" <<'PY'
import json, sys
path = sys.argv[1]
api_key = sys.argv[2]

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

data['auth_mode'] = 'apikey'
data['OPENAI_API_KEY'] = api_key
data['tokens'] = None
data['last_refresh'] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write('\n')
PY

  local tmp_file
  tmp_file="$(mktemp)"
  awk -v provider="$provider_name" -v url="$base_url" '
    BEGIN { in_rightcode = 0; provider_set = 0 }
    {
      if ($0 ~ /^\[model_providers\.rightcode\]/) {
        in_rightcode = 1
        next
      }

      if (in_rightcode == 1) {
        if ($0 ~ /^\[/) {
          in_rightcode = 0
        } else {
          next
        }
      }

      if ($0 ~ /^model_provider[[:space:]]*=/) {
        print "model_provider = \"" provider "\""
        provider_set = 1
        next
      }

      print
    }
    END {
      if (provider_set == 0) {
        print "model_provider = \"" provider "\""
      }
      print ""
      print "[model_providers." provider "]"
      print "name = \"" provider "\""
      print "base_url = \"" url "\""
      print "wire_api = \"responses\""
      print "requires_openai_auth = true"
    }
  ' "$CONFIG_FILE" > "$tmp_file"
  mv "$tmp_file" "$CONFIG_FILE"

  chmod 600 "$AUTH_FILE" "$CONFIG_FILE" || true
}

ensure_provider_name() {
  local name="$1"
  if [[ -z "$name" || ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] Invalid provider name: $name" >&2
    echo "[INFO] Allowed chars: letters, digits, ., _, -" >&2
    exit 2
  fi
}

save_named_provider_profile() {
  local name="$1"
  ensure_provider_name "$name"
  local target="${PROVIDER_PROFILE_DIR}/${name}"
  mkdir -p "$target"
  cp -f "${AUTH_FILE}" "${target}/auth.json"
  cp -f "${CONFIG_FILE}" "${target}/config.toml"
  chmod 600 "${target}/auth.json" "${target}/config.toml" || true
  echo "[OK] Saved provider profile: ${name}"
}

restore_named_provider_profile() {
  local name="$1"
  ensure_provider_name "$name"
  local target="${PROVIDER_PROFILE_DIR}/${name}"
  if [[ ! -f "${target}/auth.json" || ! -f "${target}/config.toml" ]]; then
    echo "[ERROR] Provider profile not found: ${name}" >&2
    exit 2
  fi
  cp -f "${target}/auth.json" "${AUTH_FILE}"
  cp -f "${target}/config.toml" "${CONFIG_FILE}"
  chmod 600 "${AUTH_FILE}" "${CONFIG_FILE}" || true
  echo "[OK] Switched to provider profile: ${name}"
}

list_provider_profiles() {
  local found=0
  for dir in "${PROVIDER_PROFILE_DIR}"/*; do
    [[ -d "$dir" ]] || continue
    basename "$dir"
    found=1
  done
  [[ $found -eq 1 ]] || echo "<none>"
}

auth_has_chatgpt_tokens() {
  python3 - "$AUTH_FILE" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
ok = (str(data.get('auth_mode', '')).lower() == 'chatgpt' and bool(data.get('tokens')))
print('yes' if ok else 'no')
PY
}

normalize_config_for_mode() {
  local mode="$1"
  local tmp_file
  tmp_file="$(mktemp)"

  if [[ "$mode" == "gpt" ]]; then
    awk '
      BEGIN { in_rightcode = 0; provider_set = 0 }
      {
        if ($0 ~ /^\[model_providers\.rightcode\]/) {
          in_rightcode = 1
          next
        }
        if (in_rightcode == 1) {
          if ($0 ~ /^\[/) {
            in_rightcode = 0
          } else {
            next
          }
        }
        if ($0 ~ /^model_provider[[:space:]]*=/) {
          print "model_provider = \"openai\""
          provider_set = 1
          next
        }
        print
      }
      END {
        if (provider_set == 0) {
          print "model_provider = \"openai\""
        }
      }
    ' "$CONFIG_FILE" > "$tmp_file"
    mv "$tmp_file" "$CONFIG_FILE"
  fi
}

save_profile() {
  local name="$1"
  cp -f "${AUTH_FILE}" "${PROFILE_DIR}/auth.${name}.json"
  cp -f "${CONFIG_FILE}" "${PROFILE_DIR}/config.${name}.toml"
  chmod 600 "${PROFILE_DIR}/auth.${name}.json" "${PROFILE_DIR}/config.${name}.toml" || true
  echo "[OK] Saved profile: ${name}"
}

guarded_save_profile() {
  local expected="$1"
  local force_flag="${2:-}"
  local current
  current="$(current_mode_label)"

  if [[ "${force_flag}" != "--force" && "${current}" != "${expected}" ]]; then
    echo "[ERROR] Refuse to save ${expected} while current mode is ${current}." >&2
    echo "[INFO] Switch first: codex-auth-switch ${expected}" >&2
    echo "[INFO] Or override intentionally: codex-auth-switch save-${expected} --force" >&2
    exit 2
  fi

  save_profile "${expected}"
  capture_history_snapshot "${expected}"
}

restore_profile() {
  local name="$1"
  cp -f "${PROFILE_DIR}/auth.${name}.json" "${AUTH_FILE}"
  cp -f "${PROFILE_DIR}/config.${name}.toml" "${CONFIG_FILE}"
  chmod 600 "${AUTH_FILE}" "${CONFIG_FILE}" || true
  echo "[OK] Switched to profile: ${name}"
}

show_status() {
  python3 - "$AUTH_FILE" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
mode = data.get('auth_mode')
print(f'auth_mode: {mode}')
print('has_OPENAI_API_KEY:', 'OPENAI_API_KEY' in data)
print('has_tokens:', bool(data.get('tokens')))
PY

  local current_provider=""

  if grep -q '^model_provider\s*=\s*"' "${CONFIG_FILE}"; then
    grep -E '^model_provider\s*=\s*"' "${CONFIG_FILE}" | head -n 1
    current_provider="$(sed -n 's/^model_provider[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${CONFIG_FILE}" | head -n 1)"
  else
    echo "model_provider: <not set>"
  fi

  if [[ -n "${current_provider}" ]]; then
    local current_base_url=""
    current_base_url="$(awk -v p="${current_provider}" '
      $0 ~ "^\\[model_providers\\." p "\\]" { in_section = 1; next }
      in_section == 1 && $0 ~ /^\[/ { in_section = 0 }
      in_section == 1 && $0 ~ /^base_url[[:space:]]*=[[:space:]]*"/ {
        if (match($0, /"[^"]+"/)) {
          print substr($0, RSTART, RLENGTH)
        }
        exit
      }
    ' "${CONFIG_FILE}")"
    if [[ -n "${current_base_url}" ]]; then
      echo "base_url = ${current_base_url}"
    else
      echo "base_url: <not set>"
    fi
  elif grep -q '^base_url\s*=\s*"' "${CONFIG_FILE}"; then
    grep -E '^base_url\s*=\s*"' "${CONFIG_FILE}" | head -n 1
  else
    echo "base_url: <not set>"
  fi

  [[ -f "${PROFILE_DIR}/auth.api.json" ]] && echo "profile api: yes" || echo "profile api: no"
  [[ -f "${PROFILE_DIR}/auth.gpt.json" ]] && echo "profile gpt: yes" || echo "profile gpt: no"
  [[ -f "${HISTORY_DIR}/api/session_index.jsonl" ]] && echo "history api: yes" || echo "history api: no"
  [[ -f "${HISTORY_DIR}/gpt/session_index.jsonl" ]] && echo "history gpt: yes" || echo "history gpt: no"
  echo "provider profiles:"
  list_provider_profiles | sed 's/^/  - /'
}

main() {
  ensure_files
  local cmd="${1:-status}"
  case "$cmd" in
    status)
      show_status
      ;;
    save-api)
      guarded_save_profile "api" "${2:-}"
      ;;
    save-gpt)
      guarded_save_profile "gpt" "${2:-}"
      ;;
    save-provider)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] Usage: codex-auth-switch save-provider <provider_name>" >&2
        exit 2
      fi
      save_named_provider_profile "$2"
      capture_history_snapshot "$2"
      ;;
    use-provider)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] Usage: codex-auth-switch use-provider <provider_name>" >&2
        exit 2
      fi
      capture_history_snapshot "$(current_mode_label)"
      restore_named_provider_profile "$2"
      sync_history_from_snapshots
      capture_history_snapshot "$2"
      ;;
    list-providers)
      list_provider_profiles
      ;;
    set-api-provider)
      if [[ $# -eq 3 ]]; then
        # Backward compatible mode: default provider name is api.
        set_api_provider_credentials "api" "$2" "$3"
        save_named_provider_profile "api"
        save_profile "api"
        capture_history_snapshot "api"
      elif [[ $# -ge 4 ]]; then
        set_api_provider_credentials "$2" "$3" "$4"
        save_named_provider_profile "$2"
        if [[ "$2" == "api" ]]; then
          save_profile "api"
        fi
        capture_history_snapshot "$2"
      else
        echo "[ERROR] Usage: codex-auth-switch set-api-provider [provider_name] <base_url> <api_key>" >&2
        exit 2
      fi
      echo "[OK] API provider updated and saved"
      ;;
    api)
      capture_history_snapshot "$(current_mode_label)"
      if [[ -f "${PROFILE_DIR}/auth.api.json" && -f "${PROFILE_DIR}/config.api.toml" ]]; then
        restore_profile "api"
      else
        set_auth_mode_fallback "apikey"
        chmod 600 "${AUTH_FILE}" || true
        echo "[WARN] API profile missing; only set auth_mode=apikey in auth.json"
      fi
      sync_history_from_snapshots
      capture_history_snapshot "api"
      ;;
    gpt)
      capture_history_snapshot "$(current_mode_label)"
      if [[ -f "${PROFILE_DIR}/auth.gpt.json" && -f "${PROFILE_DIR}/config.gpt.toml" ]]; then
        restore_profile "gpt"
        normalize_config_for_mode "gpt"
      else
        if [[ "$(auth_has_chatgpt_tokens)" == "yes" ]]; then
          normalize_config_for_mode "gpt"
          save_profile "gpt"
          echo "[OK] GPT profile auto-created from current ChatGPT login"
        else
          set_auth_mode_fallback "chatgpt"
          normalize_config_for_mode "gpt"
          chmod 600 "${AUTH_FILE}" "${CONFIG_FILE}" || true
          echo "[WARN] GPT profile missing; set auth_mode=chatgpt and model_provider=openai"
          echo "[INFO] Please sign in with ChatGPT in VS Code/Codex, then run: codex-auth-switch save-gpt"
        fi
      fi
      sync_history_from_snapshots
      capture_history_snapshot "gpt"
      ;;
    sync-history)
      sync_history_from_snapshots
      capture_history_snapshot "$(current_mode_label)"
      echo "[OK] Local history merged"
      ;;
    history)
      show_history "${2:-20}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "[ERROR] Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
```
