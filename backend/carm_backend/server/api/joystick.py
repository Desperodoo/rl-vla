"""
手柄遥操 HTTP API 接口
"""
import json
import time
import server
import logging
from functools import wraps
from flask import Response, stream_with_context
from basic.joystick import get_joystick_controller, DEPENDENCIES_AVAILABLE, pose_to_dict, button_to_dict

# ==========================================
# 装饰器和辅助函数
# ==========================================

def require_joystick_available(f):
    """装饰器：检查手柄遥操依赖和可用性"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not DEPENDENCIES_AVAILABLE:
            return server.failed('手柄遥操依赖模块不可用')
        
        controller = get_joystick_controller()
        if not controller.available:
            return server.failed('手柄遥操功能不可用')
        
        return f(controller, *args, **kwargs)
    return wrapper

# 统一处理 OPTIONS 请求
@server.app.before_request
def handle_options():
    """统一处理 CORS OPTIONS 请求"""
    if server.request.method == 'OPTIONS':
        rsp = server.successed()
        rsp.headers['Access-Control-Allow-Origin'] = '*'
        rsp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        rsp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return rsp

# ==========================================
# HTTP API 接口
# ==========================================

@server.app.route('/api/joystick/scan', methods=['POST', 'OPTIONS'])
@require_joystick_available
def scan_devices(controller):
    """扫描 BLE 设备"""
    try:
        devices = controller.scan_devices()
        # 如果扫描失败（返回空列表），检查是否有错误信息
        if not devices:
            error_msg = controller.get_last_error()
            if error_msg:
                logging.info(f"[API] Scan devices failed: {error_msg}")
                return server.failed(error_msg)
        
        return server.successed(body={'data': devices})
    except Exception as e:
        error_msg = f'扫描设备失败: {str(e)}'
        logging.exception(e)
        return server.failed(error_msg)

@server.app.route('/api/joystick/connect', methods=['POST', 'OPTIONS'])
@require_joystick_available
def connect_device(controller):
    """连接 BLE 设备"""
    try:
        # 使用 silent=True 防止因 Content-Type 不是 application/json 而报错
        body = server.request.get_json(silent=True) or {}
        address = body.get('address')
        name = body.get('name')  # 获取设备名称（可选）
        
        if not address:
            return server.failed('设备地址不能为空')
        
        # 如果前端没有传 name，尝试从扫描结果中查找
        if not name or name == 'Unknown':
            with controller.state.status_lock:
                scan_results = controller.state.scan_results
            # 从扫描结果中查找匹配的设备名称
            for device in scan_results:
                if device.get('address') == address:
                    name = device.get('name', 'Unknown')
                    logging.info(f"[API] Found device name from scan results: {name} ({address})")
                    break
            else:
                # 如果扫描结果中也没有找到，使用默认值
                name = name or 'Unknown'
                logging.warning(f"[API] Device name not found in scan results, using default: {name}")
        
        success = controller.connect_device(address, name)
        if success:
            return server.successed('连接成功')
        else:
            # 获取详细错误信息
            error_msg = controller.get_last_error() or '连接失败，请检查设备是否在范围内或设备是否已连接'
            logging.info(f"[API] Connect device failed: {error_msg}")
            return server.failed(error_msg)
    except Exception as e:
        error_msg = f'连接设备失败: {str(e)}'
        logging.exception(e)
        return server.failed(error_msg)

@server.app.route('/api/joystick/disconnect', methods=['POST', 'GET', 'OPTIONS'])
@require_joystick_available
def disconnect_device(controller):
    """断开 BLE 并停止遥操"""
    try:
        if server.request.method == 'GET':
            reason = server.request.args.get('reason', 'unknown')
            logging.info(f"[API] GET disconnect requested, reason={reason}")
        else:
            body = server.request.get_json(silent=True) or {}
            reason = body.get('reason', 'manual')
            logging.info(f"[API] POST disconnect requested, reason={reason}")
        
        success = controller.stop_session()
        if success:
            return server.successed(f'已停止手柄遥操 ({reason})')
        else:
            return server.failed('停止手柄遥操失败')
    except Exception as e:
        logging.exception(e)
        return server.failed('停止手柄遥操异常')

@server.app.route('/api/joystick/init_arm', methods=['POST', 'OPTIONS'])
@require_joystick_available
def init_arm(controller):
    """初始化机械臂"""
    try:
        body = server.request.get_json(silent=True) or {}
        ip = body.get('ip', '10.42.0.101')
        
        logging.info(f"[API] init_arm called with IP: {ip}")
        
        print(f"[API] Calling controller.init_arm({ip})...")
        success = controller.init_arm(ip)
        
        if success:
            print("[API] Arm initialization succeeded")
            return server.successed('初始化成功')
        else:
            # 获取更详细的错误信息
            error_msg = '初始化失败'
            if controller.controller:
                if controller.controller.carm_:
                    connected = controller.controller.carm_.is_connected()
                    print(f"[API] Arm connection status: {connected}")
                    if not connected:
                        error_msg = f'机械臂连接失败 (IP: {ip})，请检查 IP 地址和网络连接'
                else:
                    error_msg = f'无法创建机械臂连接对象 (IP: {ip})，请检查网络和机械臂状态'
            else:
                error_msg = '控制器未创建，请检查后端日志'
            
            print(f"[API] Arm initialization failed: {error_msg}")
            return server.failed(error_msg)
    except Exception as e:
        error_msg = f'初始化机械臂失败: {e}'
        logging.exception(e)
        return server.failed(error_msg)

@server.app.route('/api/joystick/calibration', methods=['POST', 'OPTIONS'])
@require_joystick_available
def set_calibration(controller):
    """设置标定模式"""
    try:
        body = server.request.get_json(silent=True) or {}
        value = body.get('value')
        
        if value is None or value < 0 or value > 3:
            return server.failed('标定值必须在 0-3 之间')
        
        success = controller.set_calibration(value)
        if success:
            return server.successed('设置成功')
        else:
            return server.failed('设置失败')
    except Exception as e:
        logging.exception(e)
        return server.failed('设置标定失败')

@server.app.route('/api/joystick/scale', methods=['POST', 'OPTIONS'])
@require_joystick_available
def set_scale(controller):
    """设置灵敏度"""
    try:
        body = server.request.get_json(silent=True) or {}
        value = body.get('value')
        
        if value is None or value < 0.1 or value > 1.0:
            return server.failed('灵敏度值必须在 0.1-1.0 之间')
        
        success = controller.set_scale(value)
        if success:
            return server.successed('设置成功')
        else:
            return server.failed('设置失败')
    except Exception as e:
        logging.exception(e)
        return server.failed('设置灵敏度失败')

@server.app.route('/api/joystick/status', methods=['GET'])
def get_status():
    """获取当前状态"""
    try:
        if not DEPENDENCIES_AVAILABLE:
            return server.successed(body={
                'data': {
                    'available': False,
                    'status': 'dependencies_not_available'
                }
            })
        
        controller = get_joystick_controller()
        if not controller.available:
            return server.successed(body={
                'data': {
                    'available': False,
                    'status': 'not_available'
                }
            })
        
        status = controller.get_status()
        return server.successed(body={'data': status})
    except Exception as e:
        logging.exception(e)
        return server.failed('获取状态失败')


@server.app.route('/api/joystick/confirm_init_position', methods=['POST', 'OPTIONS'])
@require_joystick_available
def confirm_init_position(controller):
    """确认初始化位置"""
    try:
        body = server.request.get_json(silent=True) or {}
        ip = body.get('ip', '10.42.0.101')  # 默认 IP
        
        logging.info(f"[API] confirm_init_position called with IP: {ip}")
        
        success = controller.confirm_init_position(ip)
        if success:
            return server.successed('确认初始化位置成功')
        else:
            return server.failed('确认初始化位置失败')
    except Exception as e:
        logging.exception(e)
        return server.failed('确认初始化位置失败')

@server.app.route('/api/joystick/get_init_position', methods=['GET', 'OPTIONS'])
@require_joystick_available
def get_init_position(controller):
    """获取初始化位置"""
    try:
        position = controller.get_init_position()
        # 检查是否有错误信息
        if 'error' in position and position['error']:
            error_msg = position.get('error', '获取初始化位置失败')
            logging.error(f"[API] Get init position failed: {error_msg}")
            return server.failed(error_msg)
        return server.successed(body={'data': position})
    except Exception as e:
        logging.exception(e)
        return server.failed('获取初始化位置失败')  

@server.app.route('/api/joystick/get_current_joints', methods=['GET', 'OPTIONS'])
@require_joystick_available
def get_current_joints(controller):
    """获取当前关节值"""
    try:
        # GET 请求从 query 参数获取 ip
        ip = server.request.args.get('ip', '10.42.0.101')
        logging.info(f"[API] get_current_joints called with IP: {ip}")
        
        joints = controller.get_current_joints(ip)
        # 优先检查错误信息
        if 'error' in joints and joints.get('error'):
            error_msg = joints.get('error', '获取当前关节值失败')
            logging.error(f"[API] Get current joints failed: {error_msg}")
            return server.failed(error_msg)
        # 检查是否有有效的关节值
        if joints.get('current_joints') is not None:
            # 返回格式与前端期望一致：{joints: number[]}
            return server.successed(body={'data': {'joints': joints.get('current_joints')}})
        else:
            # 没有错误信息但也没有关节值（不应该发生）
            error_msg = '获取当前关节值失败：未知错误'
            logging.error(f"[API] Get current joints failed: {error_msg}")
            return server.failed(error_msg)
    except Exception as e:
        logging.exception(e)
        return server.failed('获取当前关节值失败: ' + str(e))

@server.app.route('/api/joystick/teleop_target', methods=['GET'])
@require_joystick_available
def get_teleop_target(controller):
    """获取当前遥操作目标位姿（供数据采集端使用）"""
    try:
        state = controller.get_teleop_state()
        return server.successed(body={'data': state})
    except Exception as e:
        logging.exception(e)
        return server.failed('获取遥操作目标失败')


@server.app.route('/api/joystick/events', methods=['GET', 'OPTIONS'])
@require_joystick_available
def joystick_events(controller):
    """SSE：推送手柄位置和按钮数据（简化版：移除 data_lock）"""
    try:
        import uuid
        connection_id = str(uuid.uuid4())[:8]
        client_ip = server.request.environ.get('REMOTE_ADDR', 'unknown')
        
        logging.info(f"[SSE-{connection_id}] New connection from {client_ip}")
        
        def event_stream():
            last_sequence = -1
            last_heartbeat = time.time()
            
            try:
                while True:
                    # ✅ 直接读取数据（不加锁，参考 PyQt）
                    sequence = controller.state.data_sequence
                    pose = controller.state.pose_data
                    button = controller.state.button_data
                    battery_level = controller.state.battery_level
                    rssi = controller.state.rssi
                    gripper_pose = controller.state.gripper_pose
                    gripper_tau = controller.state.gripper_tau
                    
                    # 只对状态信息加锁（低频读取）
                    with controller.state.status_lock:
                        ble_connected = controller.state.ble_connected
                        status = controller.state.status
                        connected_address = controller.state.connected_address
                        connected_name = controller.state.connected_name
                    
                    # 检查 BLE 连接状态
                    if not ble_connected:
                        logging.info(f"[SSE-{connection_id}] BLE disconnected, stopping SSE")
                        return
                    
                    # ✅ 推送数据（包含电量、RSSI、夹爪、设备信息）
                    if sequence != last_sequence:
                        # 转换为可序列化的格式
                        pose_dict = pose_to_dict(pose) if pose else None
                        button_dict = button_to_dict(button) if button else None
                        
                        payload = {
                            "sequence": sequence,
                            "pose": pose_dict,
                            "button": button_dict,
                            "gripper_pose": gripper_pose,
                            "gripper_tau": gripper_tau,
                            "status": {
                                "ble_connected": ble_connected,
                                "status": status,
                                "battery_level": battery_level,
                                "rssi": rssi
                            },
                            "device": {
                                "address": connected_address,
                                "name": connected_name
                            },
                            "timestamp": time.time(),
                        }
                        
                        try:
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logging.error(f"[SSE-{connection_id}] Yield failed: {e}")
                            return
                        
                        last_sequence = sequence
                        last_heartbeat = time.time()
                    
                    # ✅ 心跳（3 秒间隔，包含设备信息）
                    elif time.time() - last_heartbeat > 3:
                        heartbeat_payload = {
                            "sequence": sequence,
                            "status": {
                                "ble_connected": ble_connected,
                                "status": status,
                                "battery_level": battery_level,
                                "rssi": rssi
                            },
                            "device": {
                                "address": connected_address,
                                "name": connected_name
                            },
                            "timestamp": time.time(),
                            "heartbeat": True
                        }
                        
                        try:
                            yield f"data: {json.dumps(heartbeat_payload, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            logging.error(f"[SSE-{connection_id}] Heartbeat failed: {e}")
                            return
                        
                        last_heartbeat = time.time()
                    
                    time.sleep(0.05)  # ✅ 50ms = 20Hz
                    
            except GeneratorExit:
                logging.info(f"[SSE-{connection_id}] Connection closed")
                return
            except Exception as e:
                logging.error(f"[SSE-{connection_id}] Error: {e}")
                logging.exception(e)
                return
        
        response = server.Response(event_stream(), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        logging.exception(e)
        return server.failed('创建数据流失败')


