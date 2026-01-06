/**
 * RTC 伺服主程序
 * 
 * 500Hz 控制循环:
 * 1. 从共享内存读取关键帧
 * 2. 三次样条插值
 * 3. 安全限制 (EMA + 速率限制)
 * 4. 发送到 arx5 机械臂
 * 
 * Phase A: 简单模式 (直接替换轨迹)
 * Phase B: RTC 模式 (committed/editable + blend)
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <cstring>
#include <getopt.h>

// RTC 模块
#include "../shared/shm_protocol.hpp"
#include "trajectory_buffer.hpp"
#include "safety_limiter.hpp"

// ARX5 SDK
#include "app/joint_controller.h"

namespace {
    std::atomic<bool> g_running{true};
    
    void signal_handler(int sig) {
        std::cout << "\n[Servo] 收到信号 " << sig << ", 正在退出..." << std::endl;
        g_running = false;
    }
}

/**
 * 配置
 */
struct ServoConfig {
    // 机器人
    std::string model = "X5";
    std::string interface = "can0";
    
    // 控制参数
    double servo_rate = 500.0;       // Hz
    double servo_dt = 1.0 / 500.0;   // 2ms
    
    // RTC 参数
    bool use_rtc_mode = false;       // Phase B: 启用 RTC 模式
    double commit_time = 0.10;       // 100ms
    double blend_window = 0.03;      // 30ms
    
    // 安全参数
    double ema_alpha = 0.3;
    double max_joint_delta = 0.1;    // rad/step
    double max_gripper_delta = 0.001;  // m/step
    
    // 超时
    double timeout_warning = 0.10;   // 100ms
    double timeout_hold = 0.15;      // 150ms
    
    // 调试
    bool verbose = false;
    bool dry_run = false;            // 不连接真机
};

/**
 * 伺服控制器
 */
class ServoController {
public:
    static constexpr int DOF = 7;
    
    ServoController(const ServoConfig& config) : _config(config) {
        // 配置轨迹缓冲区
        rtc::TrajectoryBufferConfig traj_config;
        traj_config.commit_time = config.commit_time;
        traj_config.blend_window = config.blend_window;
        traj_config.dof = DOF;
        _trajectory_buffer.set_config(traj_config);
        
        // 配置安全限制器
        rtc::SafetyConfig safety_config;
        safety_config.ema_alpha = config.ema_alpha;
        safety_config.max_joint_delta = config.max_joint_delta;
        safety_config.max_gripper_delta = config.max_gripper_delta;
        _safety_limiter.set_config(safety_config);
    }
    
    bool initialize() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "RTC 伺服控制器 (Phase A)" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // 1. 连接共享内存
        std::cout << "\n[1] 连接共享内存..." << std::endl;
        if (!_shm_reader.connect(rtc::ShmKeyframeReader::DEFAULT_SHM_NAME, 30.0)) {
            std::cerr << "[错误] 无法连接共享内存，请先启动 Python 推理进程" << std::endl;
            return false;
        }
        
        // 2. 初始化机械臂
        std::cout << "\n[2] 初始化机械臂..." << std::endl;
        if (!_config.dry_run) {
            try {
                _robot = std::make_unique<arx::Arx5JointController>(
                    _config.model, _config.interface
                );
                
                std::cout << "  模型: " << _config.model << std::endl;
                std::cout << "  接口: " << _config.interface << std::endl;
                
                // 获取初始状态
                auto state = _robot->get_state();
                for (int i = 0; i < 6; ++i) {
                    _current_pos[i] = state.pos()[i];
                }
                _current_pos[6] = state.gripper_pos;
                
                std::cout << "  初始位置: [";
                for (int i = 0; i < DOF; ++i) {
                    std::cout << _current_pos[i];
                    if (i < DOF - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                // 初始化安全限制器
                _safety_limiter.reset(_current_pos);
                
            } catch (const std::exception& e) {
                std::cerr << "[错误] 初始化机械臂失败: " << e.what() << std::endl;
                return false;
            }
        } else {
            std::cout << "  [模拟模式] 不连接真机" << std::endl;
            std::fill(_current_pos, _current_pos + DOF, 0.0);
            _safety_limiter.reset(_current_pos);
        }
        
        std::cout << "\n[3] 控制参数:" << std::endl;
        std::cout << "  伺服频率: " << _config.servo_rate << " Hz" << std::endl;
        std::cout << "  RTC 模式: " << (_config.use_rtc_mode ? "启用" : "禁用") << std::endl;
        std::cout << "  EMA alpha: " << _config.ema_alpha << std::endl;
        std::cout << "  最大关节速率: " << _config.max_joint_delta << " rad/step" << std::endl;
        
        std::cout << "\n✓ 伺服控制器初始化完成" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        return true;
    }
    
    void run() {
        using namespace std::chrono;
        
        std::cout << "\n[Servo] 500Hz 控制循环启动..." << std::endl;
        
        auto next_tick = steady_clock::now();
        auto last_print = steady_clock::now();
        uint64_t loop_count = 0;
        uint64_t update_count = 0;
        
        while (g_running) {
            // 1. 从共享内存读取关键帧
            bool new_data = read_keyframes();
            if (new_data) {
                update_count++;
            }
            
            // 2. 采样轨迹
            double t_now = rtc::get_monotonic_time();
            double target[DOF];
            bool has_target = sample_trajectory(t_now, target);
            
            // 3. 安全限制 + 发送命令
            if (has_target) {
                double safe_target[DOF];
                _safety_limiter.apply(target, safe_target, _current_pos);
                send_command(safe_target);
                std::copy(safe_target, safe_target + DOF, _current_pos);
            } else {
                // 没有轨迹，保持当前位置
                hold_position();
            }
            
            loop_count++;
            
            // 定期打印状态
            if (_config.verbose && duration_cast<seconds>(steady_clock::now() - last_print).count() >= 1) {
                double remaining = _trajectory_buffer.remaining_time(t_now);
                std::cout << "[Servo] loops=" << loop_count 
                          << ", updates=" << update_count
                          << ", remaining=" << remaining << "s" << std::endl;
                last_print = steady_clock::now();
            }
            
            // 4. 等待下一个 tick
            next_tick += microseconds(static_cast<int>(_config.servo_dt * 1e6));
            std::this_thread::sleep_until(next_tick);
        }
        
        std::cout << "\n[Servo] 控制循环结束" << std::endl;
        std::cout << "  总循环数: " << loop_count << std::endl;
        std::cout << "  轨迹更新数: " << update_count << std::endl;
    }
    
    void shutdown() {
        std::cout << "\n[Servo] 正在关闭..." << std::endl;
        
        if (_robot && !_config.dry_run) {
            std::cout << "  进入阻尼模式..." << std::endl;
            _robot->set_to_damping();
        }
        
        _shm_reader.disconnect();
        std::cout << "  已断开共享内存" << std::endl;
    }
    
private:
    bool read_keyframes() {
        rtc::KeyframeData data;
        if (!_shm_reader.read(data)) {
            return false;
        }
        
        // 更新轨迹缓冲区
        double t_now = rtc::get_monotonic_time();
        
        if (_config.use_rtc_mode) {
            _trajectory_buffer.update_rtc(
                data.q_key, data.H, data.dt_key, data.t_write_mono, t_now);
        } else {
            _trajectory_buffer.update_simple(
                data.q_key, data.H, data.dt_key, data.t_write_mono);
        }
        
        if (_config.verbose) {
            std::cout << "[Servo] 收到关键帧 v=" << data.version 
                      << ", H=" << data.H << std::endl;
        }
        
        return true;
    }
    
    bool sample_trajectory(double t_now, double* target) {
        return _trajectory_buffer.sample(t_now, target);
    }
    
    void send_command(const double* target) {
        if (_config.dry_run || !_robot) {
            return;
        }
        
        // 构建关节命令
        arx::JointState cmd(_robot->get_robot_config().joint_dof);
        for (int i = 0; i < 6; ++i) {
            cmd.pos()[i] = target[i];
        }
        cmd.gripper_pos = target[6];
        
        // 发送命令
        _robot->set_joint_cmd(cmd);
        _robot->send_recv_once();
        
        // 更新当前状态
        auto state = _robot->get_state();
        for (int i = 0; i < 6; ++i) {
            _current_pos[i] = state.pos()[i];
        }
        _current_pos[6] = state.gripper_pos;
    }
    
    void hold_position() {
        if (_config.dry_run || !_robot) {
            return;
        }
        
        // 发送当前位置作为目标 (保持)
        arx::JointState cmd(_robot->get_robot_config().joint_dof);
        for (int i = 0; i < 6; ++i) {
            cmd.pos()[i] = _current_pos[i];
        }
        cmd.gripper_pos = _current_pos[6];
        
        _robot->set_joint_cmd(cmd);
        _robot->send_recv_once();
    }
    
private:
    ServoConfig _config;
    
    // 共享内存
    rtc::ShmKeyframeReader _shm_reader;
    
    // 轨迹缓冲区
    rtc::TrajectoryBuffer _trajectory_buffer;
    
    // 安全限制器
    rtc::SafetyLimiter _safety_limiter;
    
    // 机械臂控制器
    std::unique_ptr<arx::Arx5JointController> _robot;
    
    // 当前位置
    double _current_pos[DOF] = {0};
};


void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  -m, --model MODEL      机械臂型号 (默认: X5)\n"
              << "  -i, --interface IFACE  CAN 接口 (默认: can0)\n"
              << "  -r, --rtc              启用 RTC 模式 (Phase B)\n"
              << "  -v, --verbose          详细输出\n"
              << "  -d, --dry-run          模拟模式 (不连接真机)\n"
              << "  -h, --help             显示帮助\n"
              << std::endl;
}


int main(int argc, char** argv) {
    ServoConfig config;
    
    // 解析命令行参数
    static struct option long_options[] = {
        {"model",     required_argument, 0, 'm'},
        {"interface", required_argument, 0, 'i'},
        {"rtc",       no_argument,       0, 'r'},
        {"verbose",   no_argument,       0, 'v'},
        {"dry-run",   no_argument,       0, 'd'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "m:i:rvdh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                config.model = optarg;
                break;
            case 'i':
                config.interface = optarg;
                break;
            case 'r':
                config.use_rtc_mode = true;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'd':
                config.dry_run = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // 设置信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // 创建并运行控制器
    ServoController controller(config);
    
    if (!controller.initialize()) {
        return 1;
    }
    
    controller.run();
    controller.shutdown();
    
    return 0;
}
