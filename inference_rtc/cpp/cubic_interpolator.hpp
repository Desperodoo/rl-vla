#ifndef CUBIC_INTERPOLATOR_HPP
#define CUBIC_INTERPOLATOR_HPP

/**
 * 三次样条插值器
 * 
 * 将 30Hz 关键帧插值到 500Hz 连续轨迹
 * 使用自然三次样条 (clamped boundary, 即端点速度为 0)
 */

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace rtc {

/**
 * 单关节三次样条
 */
class CubicSpline1D {
public:
    CubicSpline1D() = default;
    
    /**
     * 从关键帧构建样条
     * @param t 时间点数组 (长度 n)
     * @param y 关节位置数组 (长度 n)
     * @param n 关键帧数量
     * @param clamped 是否使用 clamped 边界 (端点速度为 0)
     */
    void build(const double* t, const double* y, int n, bool clamped = true) {
        if (n < 2) {
            throw std::invalid_argument("At least 2 points required");
        }
        
        _n = n;
        _t.assign(t, t + n);
        _y.assign(y, y + n);
        
        // 分配系数数组
        _a.resize(n);
        _b.resize(n);
        _c.resize(n);
        _d.resize(n);
        
        // a = y
        for (int i = 0; i < n; ++i) {
            _a[i] = y[i];
        }
        
        // 计算 h[i] = t[i+1] - t[i]
        std::vector<double> h(n - 1);
        for (int i = 0; i < n - 1; ++i) {
            h[i] = t[i + 1] - t[i];
            if (h[i] <= 0) {
                throw std::invalid_argument("Time points must be strictly increasing");
            }
        }
        
        // 构建三对角系统 A * c = rhs
        // 对于自然样条: c[0] = c[n-1] = 0
        // 对于 clamped 样条: 需要不同的边界条件
        
        if (n == 2) {
            // 两点特殊处理: 线性插值
            _c[0] = _c[1] = 0;
            _b[0] = (y[1] - y[0]) / h[0];
            _d[0] = 0;
            return;
        }
        
        // 自然三次样条 (c[0] = c[n-1] = 0)
        // 解三对角系统
        
        std::vector<double> alpha(n - 1);
        for (int i = 1; i < n - 1; ++i) {
            alpha[i] = 3.0 / h[i] * (_a[i + 1] - _a[i]) 
                     - 3.0 / h[i - 1] * (_a[i] - _a[i - 1]);
        }
        
        std::vector<double> l(n), mu(n), z(n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;
        
        for (int i = 1; i < n - 1; ++i) {
            l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        
        l[n - 1] = 1.0;
        z[n - 1] = 0.0;
        _c[n - 1] = 0.0;
        
        for (int j = n - 2; j >= 0; --j) {
            _c[j] = z[j] - mu[j] * _c[j + 1];
            _b[j] = (_a[j + 1] - _a[j]) / h[j] - h[j] * (_c[j + 1] + 2.0 * _c[j]) / 3.0;
            _d[j] = (_c[j + 1] - _c[j]) / (3.0 * h[j]);
        }
    }
    
    /**
     * 在时刻 t 采样
     */
    double eval(double t_query) const {
        if (_n < 2) {
            return 0.0;
        }
        
        // 限制在有效范围内
        t_query = std::max(_t[0], std::min(_t[_n - 1], t_query));
        
        // 找到所在区间
        int i = 0;
        for (int j = 0; j < _n - 1; ++j) {
            if (t_query >= _t[j]) {
                i = j;
            }
        }
        
        double dt = t_query - _t[i];
        return _a[i] + _b[i] * dt + _c[i] * dt * dt + _d[i] * dt * dt * dt;
    }
    
    /**
     * 获取速度 (一阶导数)
     */
    double eval_velocity(double t_query) const {
        if (_n < 2) {
            return 0.0;
        }
        
        t_query = std::max(_t[0], std::min(_t[_n - 1], t_query));
        
        int i = 0;
        for (int j = 0; j < _n - 1; ++j) {
            if (t_query >= _t[j]) {
                i = j;
            }
        }
        
        double dt = t_query - _t[i];
        return _b[i] + 2.0 * _c[i] * dt + 3.0 * _d[i] * dt * dt;
    }
    
    double t_min() const { return _n > 0 ? _t[0] : 0.0; }
    double t_max() const { return _n > 0 ? _t[_n - 1] : 0.0; }
    
private:
    int _n = 0;
    std::vector<double> _t;  // 时间点
    std::vector<double> _y;  // 值
    std::vector<double> _a, _b, _c, _d;  // 样条系数
};


/**
 * 多关节三次样条插值器
 */
class CubicInterpolator {
public:
    static constexpr int MAX_DOF = 10;
    
    CubicInterpolator() = default;
    
    /**
     * 从关键帧构建插值器
     * @param q_key 关键帧数组 [H][dof]
     * @param H 关键帧数量
     * @param dof 自由度
     * @param dt_key 关键帧时间间隔
     * @param t0 起始时间 (单调时钟)
     */
    void build(const double q_key[][MAX_DOF], int H, int dof, double dt_key, double t0) {
        if (H < 2 || dof > MAX_DOF) {
            return;
        }
        
        _H = H;
        _dof = dof;
        _dt_key = dt_key;
        _t0 = t0;
        _t_end = t0 + (H - 1) * dt_key;
        
        // 构建时间数组
        std::vector<double> t(H);
        for (int i = 0; i < H; ++i) {
            t[i] = t0 + i * dt_key;
        }
        
        // 为每个关节构建样条
        for (int j = 0; j < dof; ++j) {
            std::vector<double> y(H);
            for (int i = 0; i < H; ++i) {
                y[i] = q_key[i][j];
            }
            _splines[j].build(t.data(), y.data(), H, true);
        }
        
        _valid = true;
    }
    
    /**
     * 在时刻 t 采样所有关节位置
     * @param t_query 查询时间 (单调时钟)
     * @param q_out 输出数组 (长度 >= dof)
     */
    void eval(double t_query, double* q_out) const {
        if (!_valid) {
            return;
        }
        
        for (int j = 0; j < _dof; ++j) {
            q_out[j] = _splines[j].eval(t_query);
        }
    }
    
    /**
     * 在时刻 t 采样所有关节速度
     */
    void eval_velocity(double t_query, double* v_out) const {
        if (!_valid) {
            return;
        }
        
        for (int j = 0; j < _dof; ++j) {
            v_out[j] = _splines[j].eval_velocity(t_query);
        }
    }
    
    bool valid() const { return _valid; }
    double t0() const { return _t0; }
    double t_end() const { return _t_end; }
    int dof() const { return _dof; }
    int H() const { return _H; }
    double dt_key() const { return _dt_key; }
    
    /**
     * 检查时刻是否在有效范围内
     */
    bool in_range(double t) const {
        return _valid && t >= _t0 && t <= _t_end;
    }
    
    /**
     * 获取剩余时间
     */
    double remaining_time(double t_now) const {
        if (!_valid) return 0.0;
        return std::max(0.0, _t_end - t_now);
    }
    
private:
    int _H = 0;
    int _dof = 0;
    double _dt_key = 0.0;
    double _t0 = 0.0;
    double _t_end = 0.0;
    bool _valid = false;
    
    std::array<CubicSpline1D, MAX_DOF> _splines;
};

}  // namespace rtc

#endif  // CUBIC_INTERPOLATOR_HPP
