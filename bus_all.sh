#!/bin/bash
# run_backup.sh
# 기능: 실행 시 메뉴(1: backup, 2: unzip, 3: clean)를 보여 주고
#      선택에 따라 Python 스크립트를 호출한다.

# ── 환경 변수 ──────────────────────────────────────────────
home=$(pwd)
export PYTHONPATH="$home/src:$PYTHONPATH"

# ── 메뉴 출력 및 입력 ─────────────────────────────────────
echo "무엇을 실행할까요?"
echo "  1) backup  (현재 디렉터리 ZIP 백업)"
echo "  2) unzip   (백업 ZIP 복원)"
echo "  3) clean   (remove_except: 보존 목록 제외 일괄 삭제)"
read -rp "번호를 입력하세요 [1/2/3]: " choice

case "$choice" in
    1)  py_script="./src/util_sac/sys/zips/backup_pwd.py"  ;;
    2)  py_script="./src/util_sac/sys/zips/unzip_backup.py" ;;
    3)  py_script="./src/util_sac/sys/files/remove_except.py" ;;
    *)  echo "잘못된 입력이다."; exit 1 ;;
esac

# ── Interpreter 탐색 ──────────────────────────────────────
py_interpreters=(
    "/home/sac/miniconda3/envs/pandas/bin/python"
    "/home/sac/anaconda3/bin/python"
    "$(command -v python3)"          # 시스템 python3 fallback
)

for py in "${py_interpreters[@]}"; do
    if [ -x "$py" ]; then
        py_interpreter="$py"
        break
    fi
done

# ── 실행 ─────────────────────────────────────────────────
if [ -n "$py_interpreter" ]; then
    exec "$py_interpreter" "$py_script"
else
    echo "사용 가능한 Python Interpreter를 찾을 수 없다."
    exit 127
fi
