#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/install/setup.bash"
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONUNBUFFERED=1
python3 "$SCRIPT_DIR/run_train.py" 2>&1 | tee "$SCRIPT_DIR/train_log.txt"
EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "训练程序退出，退出码: $EXIT_CODE"
echo "日志已保存到: $SCRIPT_DIR/train_log.txt"
if [ $EXIT_CODE -ne 0 ]; then
    echo "按任意键退出..."
    read -n 1 -s
fi

# 可视化
# python3 "$SCRIPT_DIR/run_train.py" --render --render_interval 10
