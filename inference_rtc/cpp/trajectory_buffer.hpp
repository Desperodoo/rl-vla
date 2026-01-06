#ifndef TRAJECTORY_BUFFER_HPP
#define TRAJECTORY_BUFFER_HPP

/**
 * 轨迹缓冲区 (Phase B: Committed/Editable)
 * 
 * 实现 Real-Time Chunking 的核心机制:
 * - committed 区域: [t_now, t_now + commit_time] 不可修改
 * - editable 区域: (t_now + commit_time, t_now + planning_horizon] 可更新
 * - blend_window: 新旧轨迹混合过渡
 */

#include "cubic_interpolator.hpp"
#include <mutex>
#include <atomic>

namespace rtc {

/**
 * 轨迹缓冲区配置
 */
struct TrajectoryBufferConfig {
    double commit_time = 0.10;        // 100ms 已承诺窗口
    double planning_horizon = 0.2667; // 8 帧 @ 30Hz
    double blend_window = 0.03;       // 30ms 混合窗口
    int dof = 7;
};


/**
 * 轨迹缓冲区
 * 
 * Phase A: 简单模式，直接用最新轨迹
 * Phase B: RTC 模式，支持 committed/editable 分离
 */
class TrajectoryBuffer {
public:
    static constexpr int MAX_DOF = 10;
    
    TrajectoryBuffer() = default;
    explicit TrajectoryBuffer(const TrajectoryBufferConfig& config) 
        : _config(config), _dof(config.dof) {}
    
    void set_config(const TrajectoryBufferConfig& config) {
        _config = config;
        _dof = config.dof;
    }
    
    /**
     * 更新轨迹 (Phase A: 直接替换)
     * @param q_key 关键帧数组 [H][dof]
     * @param H 关键帧数量
     * @param dt_key 关键帧时间间隔
     * @param t_write 写入时间 (单调时钟)
     */
    void update_simple(const double q_key[][CubicInterpolator::MAX_DOF], 
                       int H, double dt_key, double t_write) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        // 直接构建新的插值器
        // 注意: t_write 是 Python 端写入共享内存的时间
        // 我们用它作为轨迹的起始时间
        _interpolator.build(q_key, H, _dof, dt_key, t_write);
        _has_trajectory = true;
        _last_update_time = t_write;
    }
    
    /**
     * 更新轨迹 (Phase B: RTC 模式，支持 blend)
     * @param q_key 关键帧数组 [H][dof]
     * @param H 关键帧数量
     * @param dt_key 关键帧时间间隔
     * @param t_write 写入时间 (单调时钟)
     * @param t_now 当前时间 (单调时钟)
     */
    void update_rtc(const double q_key[][CubicInterpolator::MAX_DOF],
                    int H, double dt_key, double t_write, double t_now) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        // 计算边界时间
        double t_commit_end = t_now + _config.commit_time;
        double t_blend_end = t_commit_end + _config.blend_window;
        
        // 如果没有旧轨迹，直接使用新轨迹
        if (!_has_trajectory) {
            _interpolator.build(q_key, H, _dof, dt_key, t_write);
            _has_trajectory = true;
            _last_update_time = t_write;
            _blending = false;
            return;
        }
        
        // 保存旧插值器
        _old_interpolator = _interpolator;
        
        // 构建新插值器
        _interpolator.build(q_key, H, _dof, dt_key, t_write);
        
        // 启用混合
        _blending = true;
        _blend_start_time = t_commit_end;
        _blend_end_time = t_blend_end;
        
        _last_update_time = t_write;
    }
    
    /**
     * 在时刻 t 采样
     * @param t_now 当前时间 (单调时钟)
     * @param q_out 输出位置 [dof]
     * @return 是否成功采样
     */
    bool sample(double t_now, double* q_out) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        if (!_has_trajectory) {
            return false;
        }
        
        if (_blending && t_now >= _blend_start_time && t_now < _blend_end_time) {
            // 混合模式
            double alpha = (t_now - _blend_start_time) / (_blend_end_time - _blend_start_time);
            alpha = alpha * alpha * (3.0 - 2.0 * alpha);  // smoothstep
            
            double q_old[MAX_DOF], q_new[MAX_DOF];
            _old_interpolator.eval(t_now, q_old);
            _interpolator.eval(t_now, q_new);
            
            for (int i = 0; i < _dof; ++i) {
                q_out[i] = (1.0 - alpha) * q_old[i] + alpha * q_new[i];
            }
        } else {
            // 正常模式
            if (_blending && t_now >= _blend_end_time) {
                _blending = false;  // 混合完成
            }
            _interpolator.eval(t_now, q_out);
        }
        
        return true;
    }
    
    /**
     * 检查轨迹是否超时
     * @param t_now 当前时间
     * @return 超时类型: 0=正常, 1=警告, 2=保持, 3=回home
     */
    int check_timeout(double t_now) const {
        if (!_has_trajectory) {
            return 3;  // 没有轨迹，应该回 home
        }
        
        double elapsed = t_now - _last_update_time;
        
        if (elapsed > _config.planning_horizon + 0.5) {
            return 3;  // 严重超时，回 home
        } else if (elapsed > _config.planning_horizon) {
            return 2;  // 轨迹用完，保持位置
        } else if (elapsed > _config.commit_time * 1.5) {
            return 1;  // 警告，推理可能太慢
        }
        
        return 0;  // 正常
    }
    
    /**
     * 获取剩余轨迹时间
     */
    double remaining_time(double t_now) const {
        if (!_has_trajectory) {
            return 0.0;
        }
        return _interpolator.remaining_time(t_now);
    }
    
    /**
     * 检查时刻是否在轨迹有效范围内
     */
    bool in_range(double t_now) const {
        return _has_trajectory && _interpolator.in_range(t_now);
    }
    
    /**
     * 获取最后更新时间
     */
    double last_update_time() const { return _last_update_time; }
    
    bool has_trajectory() const { return _has_trajectory; }
    int dof() const { return _dof; }
    
    /**
     * 清空轨迹
     */
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _has_trajectory = false;
        _blending = false;
    }
    
private:
    TrajectoryBufferConfig _config;
    int _dof = 7;
    
    std::mutex _mutex;
    CubicInterpolator _interpolator;
    CubicInterpolator _old_interpolator;  // 用于混合
    
    bool _has_trajectory = false;
    double _last_update_time = 0.0;
    
    // 混合状态
    bool _blending = false;
    double _blend_start_time = 0.0;
    double _blend_end_time = 0.0;
};

}  // namespace rtc

#endif  // TRAJECTORY_BUFFER_HPP
