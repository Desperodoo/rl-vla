#ifndef SAFETY_LIMITER_HPP
#define SAFETY_LIMITER_HPP

/**
 * 安全限制器
 * 
 * 包含:
 * - EMA 滤波
 * - 关节速率限制
 * - 关节位置限制
 */

#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace rtc {

/**
 * EMA (指数移动平均) 滤波器
 */
template<int DOF>
class EMAFilter {
public:
    EMAFilter(double alpha = 0.3) : _alpha(alpha), _initialized(false) {}
    
    void set_alpha(double alpha) {
        _alpha = std::max(0.0, std::min(1.0, alpha));
    }
    
    void reset() {
        _initialized = false;
    }
    
    void reset(const double* initial_value) {
        for (int i = 0; i < DOF; ++i) {
            _state[i] = initial_value[i];
        }
        _initialized = true;
    }
    
    void filter(const double* input, double* output) {
        if (!_initialized) {
            for (int i = 0; i < DOF; ++i) {
                _state[i] = input[i];
            }
            _initialized = true;
        }
        
        for (int i = 0; i < DOF; ++i) {
            _state[i] = _alpha * input[i] + (1.0 - _alpha) * _state[i];
            output[i] = _state[i];
        }
    }
    
    const double* state() const { return _state.data(); }
    
private:
    double _alpha;
    bool _initialized;
    std::array<double, DOF> _state;
};


/**
 * 安全限制器配置
 */
struct SafetyConfig {
    // 关节位置限制 (rad)
    std::array<double, 7> joint_pos_min = {-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0};
    std::array<double, 7> joint_pos_max = {3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.08};
    
    // 关节速率限制 (rad/step for joints, m/step for gripper)
    double max_joint_delta = 0.1;   // 0.1 rad/step @ 500Hz = ~50 rad/s
    double max_gripper_delta = 0.001;  // 0.001 m/step @ 500Hz = 0.5 m/s
    
    // EMA 滤波系数
    double ema_alpha = 0.3;
    
    // 超时设置 (秒)
    double timeout_warning = 0.1;   // 100ms 无更新警告
    double timeout_hold = 0.15;     // 150ms 无更新保持位置
    double timeout_home = 1.0;      // 1s 无更新回 home
};


/**
 * 安全限制器
 */
class SafetyLimiter {
public:
    static constexpr int DOF = 7;  // 6 关节 + 1 夹爪
    
    SafetyLimiter() = default;
    explicit SafetyLimiter(const SafetyConfig& config) : _config(config) {
        _ema_filter.set_alpha(config.ema_alpha);
    }
    
    void set_config(const SafetyConfig& config) {
        _config = config;
        _ema_filter.set_alpha(config.ema_alpha);
    }
    
    /**
     * 应用安全限制
     * @param input 原始输入 [DOF]
     * @param output 安全输出 [DOF]
     * @param current 当前位置 [DOF] (用于速率限制)
     */
    void apply(const double* input, double* output, const double* current) {
        // 1. 复制输入
        std::array<double, DOF> safe;
        for (int i = 0; i < DOF; ++i) {
            safe[i] = input[i];
        }
        
        // 2. 应用位置限制
        for (int i = 0; i < DOF; ++i) {
            safe[i] = std::max(_config.joint_pos_min[i], 
                              std::min(_config.joint_pos_max[i], safe[i]));
        }
        
        // 3. 应用速率限制
        if (current != nullptr) {
            // 关节 (0-5)
            for (int i = 0; i < 6; ++i) {
                double delta = safe[i] - current[i];
                delta = std::max(-_config.max_joint_delta, 
                                std::min(_config.max_joint_delta, delta));
                safe[i] = current[i] + delta;
            }
            
            // 夹爪 (6)
            double gripper_delta = safe[6] - current[6];
            gripper_delta = std::max(-_config.max_gripper_delta,
                                    std::min(_config.max_gripper_delta, gripper_delta));
            safe[6] = current[6] + gripper_delta;
        }
        
        // 4. EMA 滤波
        _ema_filter.filter(safe.data(), output);
    }
    
    /**
     * 重置滤波器状态
     */
    void reset() {
        _ema_filter.reset();
    }
    
    void reset(const double* initial_state) {
        _ema_filter.reset(initial_state);
    }
    
    const SafetyConfig& config() const { return _config; }
    
private:
    SafetyConfig _config;
    EMAFilter<DOF> _ema_filter;
};

}  // namespace rtc

#endif  // SAFETY_LIMITER_HPP
