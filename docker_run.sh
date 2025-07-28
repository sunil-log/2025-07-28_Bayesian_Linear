#!/bin/bash
#
# docker_run.py의 기능을 수행하는 셸 스크립트.
# (외부 데이터 디렉터리 탐색 및 마운트 로직 제거됨)

# ── 1. 변수 설정 ────────────────────────────────────────────────
# Docker 컨테이너 내부 마운트 경로 (고정값)
IN_ENTRY="sac"

# 현재 사용자 및 작업 디렉터리
USER_NAME=$(whoami)
OUT_ENTRY=$(pwd)


# ── 2. Docker 명령어 구성 ─────────────────────────────────────────
# 배열(array)을 사용해 공백이나 특수 문자가 포함된 인자를 안전하게 처리
DOCKER_CMD=("docker" "run")

# 사용자 이름이 'sac'일 때만 -u 옵션 추가
if [ "$USER_NAME" = "sac" ]; then
    DOCKER_CMD+=("-u" "$(id -u):$(id -g)")
fi

# 공통 볼륨 마운트 및 기타 Docker 옵션 추가
DOCKER_CMD+=(
    "--rm"
    "--gpus" "all"
    "-v" "${OUT_ENTRY}:/${IN_ENTRY}"
    "-v" "/usr/share/fonts:/usr/share/fonts:ro"
    "--entrypoint" "/${IN_ENTRY}/docker_entrypoint.sh"
    "sac/lightning" # 실행할 Docker 이미지
)


# ── 3. Docker 컨테이너 실행 ───────────────────────────────────────
# exec를 사용하여 현재 셸 프로세스를 docker 명령으로 대체
exec "${DOCKER_CMD[@]}"