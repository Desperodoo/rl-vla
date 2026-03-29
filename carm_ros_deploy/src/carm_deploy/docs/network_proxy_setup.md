# 双网口网络与代理通用配置

这是一份可直接复用的配置文档，适用于：

- 一根网线用于上网
- 一根网线连接机械臂下位机
- 两台机器的网关、掩码、DNS 保持一致
- 只让上网网线负责外网流量
- 代理可用 `xx.xx.xx.xx:7890` 或 `xx.xx.xx.xx:7890`

---

## 1. 需要固定的参数

### 1.1 上网网口

- 子网掩码：`/24`
- 网关：`xx.xx.xx.254`
- DNS：`10.8.8.8`, `8.8.4.4`, `8.8.8.8`
- 机器 IP 示例：
  - 新机器：`xx.xx.xx.193/24`


### 1.2 机械臂下位机网口

- 通常使用独立网段，例如 `10.42.0.0/24`
- 示例 IP：`10.42.0.11/24`
- 示例网关：`10.42.0.1`

如果你的下位机地址不是这个值，就替换成实际地址即可。

### 1.3 代理

代理地址：

- `http://xx.xx.xx.xx:7890`


---

## 2. 配置目标

1. 上网网线负责所有外网访问；
2. 机械臂下位机网线只负责机械臂局域网通信；
3. 外网程序使用代理；
4. 避免默认路由误走机械臂网线。

---

## 3. NetworkManager 配置

下面用 `nmcli` 配置两个网口。把连接名替换成你机器上的实际名字即可。

### 3.1 上网网口


```bash
nmcli con mod "internet" \
  ipv4.method manual \
  ipv4.addresses xx.xx.xx.193/24 \
  ipv4.gateway xx.xx.xx.254 \
  ipv4.dns "10.8.8.8 8.8.4.4 8.8.8.8" \
  ipv4.ignore-auto-dns yes
```

### 3.2 机械臂下位机网口

```bash
nmcli con mod "carm" \
  ipv4.method manual \
  ipv4.addresses 10.42.0.11/24 \
  ipv4.gateway 10.42.0.1
```

如果你的下位机 IP 不同，就把 `10.42.0.11/24` 改成实际值。

---

## 4. 路由设置

如果两个网口都配置了网关，建议确保外网默认路由始终走上网网线。

### 推荐做法

给上网网线更低的 route metric，给机械臂网线更高的 route metric：

```bash
nmcli con mod "internet" ipv4.route-metric 100
nmcli con mod "carm" ipv4.route-metric 20101
```

也可以直接禁止机械臂网线成为默认出口：

```bash
nmcli con mod "carm" ipv4.never-default yes
```

### 生效方式

```bash
nmcli con down "internet"
nmcli con up "internet"

nmcli con down "carm"
nmcli con up "carm"
```

如果是远程操作，建议逐条执行，避免把自己断网。

---

## 5. 代理配置

### 5.1 当前终端临时生效

```bash
export HTTP_PROXY=http://xx.xx.xx.xx:7890
export HTTPS_PROXY=http://xx.xx.xx.xx:7890
export ALL_PROXY=http://xx.xx.xx.xx:7890
export http_proxy=http://xx.xx.xx.xx:7890
export https_proxy=http://xx.xx.xx.xx:7890
export all_proxy=http://xx.xx.xx.xx:7890
export NO_PROXY=localhost,127.0.0.1,::1,10.42.0.0/24,xx.xx.xx.0/24
export no_proxy=localhost,127.0.0.1,::1,10.42.0.0/24,xx.xx.xx.0/24
```

如果你要使用 `xx.xx.xx.xx:7890`，把上面的地址整体替换即可。

### 5.2 登录后自动生效

如果希望登录后自动生效，可以写到 `/etc/environment`：

```bash
http_proxy="http://xx.xx.xx.xx:7890"
https_proxy="http://xx.xx.xx.xx:7890"
HTTP_PROXY="http://xx.xx.xx.xx:7890"
HTTPS_PROXY="http://xx.xx.xx.xx:7890"
all_proxy="http://xx.xx.xx.xx:7890"
ALL_PROXY="http://xx.xx.xx.xx:7890"
no_proxy="localhost,127.0.0.1,::1,10.42.0.0/24,xx.xx.xx.0/24"
NO_PROXY="localhost,127.0.0.1,::1,10.42.0.0/24,xx.xx.xx.0/24"
```

---

## 6. 配置完成后的检查

```bash
ip -brief addr
ip route
ip route get 8.8.8.8
ip route get 10.42.0.1
nmcli device status
nmcli connection show --active
env | grep -i proxy
```

期望结果：

- `8.8.8.8` 走上网网线
- `10.42.0.1` 走机械臂网线
- 代理环境变量指向 `129` 或 `129`

---

## 7. 可直接复用的模板

### 上网网口模板

- IP：`xx.xx.xx.X/24`
- 网关：`xx.xx.xx.254`
- DNS：`10.8.8.8 8.8.4.4 8.8.8.8`
- 代理：`xx.xx.xx.xx:7890` 或 `xx.xx.xx.xx:7890`

### 机械臂网口模板

- IP：`10.42.0.X/24`
- 网关：`10.42.0.1`
- 默认不走外网

---

## 8. 常见问题

- 如果系统默认路由走到了机械臂网线，先检查 `ipv4.route-metric` 和 `ipv4.never-default`。
- 如果外网程序还是无法联网，检查当前 shell、VS Code Remote 和 `/etc/environment` 的代理是否一致。
- 如果机械臂网段被代理拦截，确认 `NO_PROXY` 已包含 `10.42.0.0/24`。
