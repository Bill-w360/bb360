#!/usr/bin/env bash
# scripts/init_router.sh
# 用法：
#   chmod +x scripts/init_router.sh
#   ./scripts/init_router.sh
# 作用：
#   1) 自动定位项目根目录（脚本所在目录的上一级）
#   2) 确保 router/ 与 extras/ 是 Python 包
#   3) 导出 PYTHONPATH（当前会话有效）
#   4) 运行一次导入自检

###  ！！！ 输入这个 source ./.router_env


set -euo pipefail

# 定位项目根目录：scripts/ 的上一级
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${THIS_DIR}/.." && pwd )"

echo "[init] PROJECT_ROOT = ${PROJECT_ROOT}"

# 确保包结构
mkdir -p "${PROJECT_ROOT}/router" "${PROJECT_ROOT}/extras"
[ -f "${PROJECT_ROOT}/router/__init__.py" ] || echo "# package" > "${PROJECT_ROOT}/router/__init__.py"
[ -f "${PROJECT_ROOT}/extras/__init__.py" ] || echo "# package" > "${PROJECT_ROOT}/extras/__init__.py"

# 导出 PYTHONPATH（当前 shell 会话）
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "[init] PYTHONPATH = ${PYTHONPATH}"

# 可选：创建 .env 文件，方便以后 `source` 使用
ENV_FILE="${PROJECT_ROOT}/.router_env"
cat > "${ENV_FILE}" <<EOF
# 加载本项目包路径
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
EOF
echo "[init] wrote ${ENV_FILE}"
echo "[init] 提示：以后可执行 'source ${ENV_FILE}' 来快速加载 PYTHONPATH"

# 自检：尝试导入
python - <<'PY'
import sys
print("[check] sys.path[0..3] =", sys.path[:3])
import router, extras
print("[check] import router, extras OK")
PY

echo "[init] 完成 ✅ 你现在可以直接运行："
echo "       python extras/make_assign_history.py --help"
echo "   或   python -m extras.make_assign_history --help"
