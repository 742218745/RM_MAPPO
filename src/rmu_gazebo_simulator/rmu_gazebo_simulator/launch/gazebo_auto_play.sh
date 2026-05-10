#!/bin/bash
# ============================================================
# 延迟自动播放 Gazebo 仿真
#
# 等待 Gazebo 完全加载后，通过 ign service 发送 play 命令
# 用法: bash gazebo_auto_play.sh [延迟秒数，默认8]
# ============================================================

DELAY=${1:-8}

echo "[auto_play] 等待 ${DELAY}s 让 Gazebo 完全加载..."
sleep "$DELAY"

# Ignition Gazebo 默认 world 名称为 "default"
SERVICE_PATH="/world/default/control"

# 等待 world_control service 出现
echo "[auto_play] 查找 service: ${SERVICE_PATH}"
for i in $(seq 1 30); do
    if ign service --list 2>/dev/null | grep -q "${SERVICE_PATH}"; then
        echo "[auto_play] 检测到 service (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[auto_play] 超时，尝试列出 service:"
        ign service --list 2>&1 | grep "control" | head -10
        echo "[auto_play] 自动播放失败，请手动点击播放按钮"
        exit 1
    fi
    sleep 1
done

# 发送 play 请求
echo "[auto_play] 发送 play 命令..."
ign service -s "${SERVICE_PATH}" \
    --reqtype ign_msgs.WorldControl \
    --reptype ign_msgs.Boolean \
    --timeout 5000 \
    --req 'pause: false' 2>&1

if [ $? -eq 0 ]; then
    echo "[auto_play] 仿真已自动开始运行"
else
    echo "[auto_play] 自动播放失败，请手动点击播放按钮"
fi
