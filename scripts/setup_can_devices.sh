#!/bin/bash
# ARX5 SDK CAN设备自动配置脚本
# 只识别真正的 CANable2 设备 (16d0:117e)
# 注意: CH341 (1a86:7523) 是普通串口，用于 UArm 舵机控制器，不是 CAN 设备
# 版本: 2024.12.18 (修正版)

set -e

echo "=========================================="
echo "  ARX5 SDK CAN 设备自动配置"
echo "  仅识别 CANable2 设备"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# 第1步：扫描所有 CANable2 设备
# ============================================================
echo -e "${BLUE}[1/4]${NC} 扫描 CANable2 设备..."
echo ""

DEVICES=()
SERIALS=()
TTYDEVS=()

# 查找所有 ttyACM 和 ttyUSB 设备
for device in /dev/ttyACM* /dev/ttyUSB*; do
    [ -c "$device" ] || continue
    
    device_info=$(udevadm info -a -n "$device" 2>/dev/null)
    
    # 提取 vendor 和 product
    vendor=$(echo "$device_info" | grep 'ATTRS{idVendor}==' | head -1 | sed 's/.*=="\([^"]*\)".*/\1/')
    product=$(echo "$device_info" | grep 'ATTRS{idProduct}==' | head -1 | sed 's/.*=="\([^"]*\)".*/\1/')
    serial=$(echo "$device_info" | grep 'ATTRS{serial}==' | head -1 | sed 's/.*=="\([^"]*\)".*/\1/')
    
    # 只检查 CANable2 (16d0:117e)
    # 注意: CH341 (1a86:7523) 是普通串口，不是 CAN 设备！
    if [ "$vendor" = "16d0" ] && [ "$product" = "117e" ]; then
        DEVICES+=("$device")
        SERIALS+=("$serial")
        TTYDEVS+=("$(basename $device)")
        echo -e "${GREEN}✓${NC} 发现 CANable2: $device"
        echo "  序列号: $serial"
    elif [ "$vendor" = "1a86" ] && [ "$product" = "7523" ]; then
        # CH341 是普通串口，用于 UArm 舵机控制器
        echo -e "${YELLOW}ℹ${NC} 发现 CH341 串口: $device (用于 UArm 舵机控制器)"
        UARM_DEVICE="$device"
        UARM_SERIAL="$serial"
    fi
done

# 为 UArm 串口创建固定的符号链接
if [ -n "$UARM_DEVICE" ]; then
    echo ""
    echo -e "${BLUE}配置 UArm 串口符号链接...${NC}"
    UARM_UDEV_RULE="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"1a86\", ATTRS{idProduct}==\"7523\", SYMLINK+=\"uarm_servo\", MODE=\"0666\""
    
    if ! grep -q "uarm_servo" /etc/udev/rules.d/99-uarm.rules 2>/dev/null; then
        echo "$UARM_UDEV_RULE" | sudo tee /etc/udev/rules.d/99-uarm.rules > /dev/null
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        echo -e "${GREEN}✓${NC} 已创建 /dev/uarm_servo 符号链接"
    else
        echo -e "${GREEN}✓${NC} UArm 串口规则已存在"
    fi
    echo "  UArm 串口: $UARM_DEVICE → /dev/uarm_servo"
fi

if [ ${#DEVICES[@]} -eq 0 ]; then
    echo ""
    echo -e "${RED}✗${NC} 未找到任何 CANable2 设备！"
    echo ""
    echo "请检查："
    echo "  1. CANable2 设备是否已连接"
    echo "  2. 运行 'lsusb' 查看 USB 设备列表"
    echo "  3. 查找 VID:PID = 16d0:117e 的设备"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 共发现 ${#DEVICES[@]} 个 CANable2 设备${NC}"
echo ""

# ============================================================
# 第2步：配置 udev 规则以固定设备名
# ============================================================
echo -e "${BLUE}[2/4]${NC} 配置 udev 规则..."
echo ""

UDEV_RULES_FILE="/etc/udev/rules.d/arx_can.rules"

# 检查是否需要更新udev规则
NEED_UPDATE=0
for i in "${!DEVICES[@]}"; do
    serial="${SERIALS[$i]}"
    
    if ! grep -q "serial.*$serial" "$UDEV_RULES_FILE" 2>/dev/null; then
        NEED_UPDATE=1
        break
    fi
done

if [ $NEED_UPDATE -eq 1 ]; then
    echo -e "${YELLOW}⚠${NC} 需要更新 udev 规则 (需要管理员权限)"
    echo ""
    
    # 创建新的udev规则内容
    UDEV_CONTENT=""
    for i in "${!DEVICES[@]}"; do
        serial="${SERIALS[$i]}"
        can_name="can$i"
        device="${DEVICES[$i]}"
        device_info=$(udevadm info -a -n "$device" 2>/dev/null)
        vendor=$(echo "$device_info" | grep 'ATTRS{idVendor}==' | head -1 | sed 's/.*=="\([^"]*\)".*/\1/')
        product=$(echo "$device_info" | grep 'ATTRS{idProduct}==' | head -1 | sed 's/.*=="\([^"]*\)".*/\1/')
        
        # 为不同的索引创建不同的符号链接
        SYMLINK="arxcan$i"
        if [ $i -eq 0 ]; then
            DESCRIPTION="ARX5 机械臂"
        else
            DESCRIPTION="ARX5 机械臂 $i"
        fi
        
        # 根据设备类型创建相应的规则
        UDEV_CONTENT+="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$vendor\", ATTRS{idProduct}==\"$product\", ATTRS{serial}==\"$serial\", SYMLINK+=\"$SYMLINK\"  # $DESCRIPTION
"
    done
    
    # 备份原有规则
    if [ -f "$UDEV_RULES_FILE" ]; then
        echo -e "${YELLOW}ℹ${NC} 备份原有规则: ${UDEV_RULES_FILE}.bak"
        sudo cp "$UDEV_RULES_FILE" "${UDEV_RULES_FILE}.bak"
    fi
    
    # 写入新规则
    echo "$UDEV_CONTENT" | sudo tee "$UDEV_RULES_FILE" > /dev/null
    
    # 重新加载udev规则
    echo "重新加载 udev 规则..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    echo -e "${GREEN}✓${NC} udev 规则已更新"
else
    echo -e "${GREEN}✓${NC} udev 规则已存在，无需更新"
fi

echo ""
echo "udev 规则内容:"
echo "---"
cat "$UDEV_RULES_FILE" 2>/dev/null || echo "(无规则)"
echo "---"
echo ""

# ============================================================
# 第3步：创建 SLCAN 虚拟网卡并启动
# ============================================================
echo -e "${BLUE}[3/4]${NC} 创建并启动 CAN 网卡..."
echo ""

for i in "${!DEVICES[@]}"; do
    device="${DEVICES[$i]}"
    can_name="can$i"
    
    echo "配置 $can_name (ARX5 机械臂):"
    echo "  物理设备: $device"
    
    # 先停止可能存在的旧进程
    sudo pkill -f "slcand.*$device" 2>/dev/null || true
    sudo ip link delete "$can_name" 2>/dev/null || true
    sleep 0.5
    
    # 使用 slcand 创建虚拟网卡
    if command -v slcand &> /dev/null; then
        echo "  → 使用 slcand 创建虚拟网卡..."
        sudo slcand -o -c -s8 "$device" "$can_name" 2>/dev/null || {
            echo -e "  ${YELLOW}⚠${NC} slcand 失败，尝试备用方案..."
        }
    else
        echo -e "  ${YELLOW}⚠${NC} slcand 未安装，尝试 slcan_attach..."
        sudo slcan_attach -f -s8 "$device" 2>/dev/null || true
        sudo ip link set slcan0 name "$can_name" 2>/dev/null || true
    fi
    
    # 启动网卡
    echo "  → 启动网卡..."
    
    # 先尝试关闭已存在的网卡
    sudo ip link set "$can_name" down 2>/dev/null || true
    
    # 启动网卡
    sudo ip link set "$can_name" up type can bitrate 1000000 2>/dev/null || {
        echo -e "  ${YELLOW}⚠${NC} 设置 bitrate 失败，直接启动..."
        sudo ip link set "$can_name" up 2>/dev/null || true
    }
    
    # 验证
    if ip link show "$can_name" 2>/dev/null | grep -q "UP"; then
        echo -e "  ${GREEN}✓${NC} $can_name 启动成功"
    else
        echo -e "  ${RED}✗${NC} $can_name 启动失败，请检查"
    fi
    echo ""
done

# ============================================================
# 第4步：验证和总结
# ============================================================
echo -e "${BLUE}[4/4]${NC} 验证配置..."
echo ""

echo -e "${BLUE}当前 CAN 接口状态:${NC}"
ip link show | grep -E "^[0-9]+.*can" || echo "  (无 CAN 接口)"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ CAN 设备配置完成${NC}"
echo "=========================================="
echo ""

echo -e "${BLUE}设备映射:${NC}"
for i in "${!DEVICES[@]}"; do
    device="${DEVICES[$i]}"
    serial="${SERIALS[$i]}"
    can_name="can$i"
    
    echo "  $can_name ← $device (ARX5 机械臂)"
    echo "         序列号: $serial"
    echo ""
done

echo -e "${BLUE}其他设备说明:${NC}"
echo "  CH341 串口 (/dev/ttyUSBx) → UArm 舵机控制器"
echo "  请在 servo_zero.py 中配置正确的串口路径"
echo ""

echo -e "${BLUE}下一步:${NC}"
echo "1. 验证 CAN 通信: python test_can.py can0"
echo "2. 测试 UArm: python src/uarm/scripts/Uarm_teleop/servo_zero.py"
echo "3. 启动遥操作: python src/uarm/scripts/Follower_Arm/ARX/arx_teleop.py"
echo ""

echo -e "${BLUE}环境设置:${NC}"
echo "  source setup_env.sh"
echo ""
