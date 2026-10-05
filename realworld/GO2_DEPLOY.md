# GO2 端部署说明

这份文档只整理 `go2` 机器人本地需要的代码和依赖。

当前架构是：

1. 远程服务器启动 vLLM
2. 远程服务器启动 `realworld/server.py`
3. GO2 本机启动 `realworld/go2_vln_client.py`

所以 GO2 端只负责：

- 订阅相机和里程计
- 把当前图像发给远程 `/eval_vln`
- 接收动作序列
- 用本地 PID 控制器执行动作

GO2 端不需要加载 Qwen 模型，也不需要启动 Flask。

## 远程 server 启动

在运行推理的远程机器上启动服务端，并显式指定监听地址。`BIND_HOST` 只是占位符，部署时替换成实际的监听接口；不要把真实地址写入仓库或提交记录。

先安装服务端依赖：

```bash
pip install -r realworld/requirements.txt
```

```bash
python realworld/server.py \
  --host BIND_HOST \
  --port 5801
```

服务端默认从本机的 M1/M2 vLLM 端点读取模型；如果端点不在本机，请额外传入 `--m1_server_url` 和 `--m2_server_url`。

## 需要拷贝的代码

如果不是整仓库部署到 GO2，而是只拷最小运行文件，则至少需要这 3 个文件：

- [go2_vln_client.py](go2_vln_client.py)
- [pid_controller.py](pid_controller.py)
- [utils.py](utils.py)

推荐目录结构：

```text
realworld/
  go2_vln_client.py
  pid_controller.py
  utils.py
```

原因：

- `go2_vln_client.py` 是 GO2 端主程序
- `pid_controller.py` 提供底盘控制的 PD/PID 逻辑
- `utils.py` 提供读写锁 `ReadWriteLock`

`go2_vln_client.py` 已经兼容两种导入方式：

- 在仓库根目录运行时，走 `from realworld.xxx import ...`
- 在 `realworld/` 目录单独拷贝运行时，走 `from xxx import ...`

## GO2 端需要的 Python 包

### 必需的通用 Python 包

- `numpy`
- `requests`
- `Pillow`

用途：

- `numpy`：处理位姿矩阵和图像数组
- `requests`：向远程 server 发 HTTP 请求
- `Pillow`：把 ROS 图像编码成 JPEG 上传

安装示例：

```bash
pip install numpy requests pillow
```

注意：

- GO2 客户端不需要 `flask`
- GO2 客户端不需要 `transformers`
- GO2 客户端不需要 `vllm`
- GO2 客户端不需要 `ms-swift`

## GO2 端需要的 ROS2 / Unitree 相关包

客户端脚本直接依赖以下模块：

- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `unitree_go.msg`
- `unitree_api.msg`

其中：

- `rclpy`：ROS2 Python 节点运行时
- `sensor_msgs.msg.Image`：相机图像消息
- `cv_bridge`：ROS 图像转 numpy
- `unitree_go.msg.SportModeState`：GO2 里程计 / 姿态
- `unitree_api.msg.Request`、`RequestHeader`：下发运动控制命令

这部分通常不是通过 `pip` 安装，而是来自：

- GO2 本机 ROS2 环境
- Unitree SDK/消息包工作区

在启动前，通常需要先 `source` 对应环境，例如：

```bash
source /opt/ros/<ros_distro>/setup.bash
source <unitree_ros2_ws>/install/setup.bash
```

具体路径按你的 GO2 环境调整。

## 客户端代码做了什么

### 1. 图像采集与上传

`go2_vln_client.py` 订阅：

- `/camera/camera/color/image_raw`
- `/sportmodestate`

收到 RGB 图像后：

- 使用 `cv_bridge` 转成 `bgr8`
- 在发 HTTP 前转成 RGB
- 编码成 JPEG
- 发给远程 `server.py` 的 `/eval_vln`

发送的 JSON 字段有：

- `reset`
- `instruction`
- `session_id`

### 2. 动作执行

服务端返回动作列表，例如：

- `[1, 1, 3]`
- `[2, 1, 1]`
- `[0]`

动作定义：

- `1`: 前进 25 cm
- `2`: 左转 15 度
- `3`: 右转 15 度
- `0`: 停止

客户端拿到动作后，会更新局部目标位姿，然后由 `pid_controller.py` 控制底盘逼近该目标。

### 3. reset 行为

客户端当前逻辑是：

- 第一次请求自动发送 `reset=True`
- 后续请求发送 `reset=False`

这样服务端就会为该 `session_id` 新建一条轨迹历史。

## GO2 端启动命令

如果你在仓库根目录启动：

```bash
python realworld/go2_vln_client.py \
  --server-url http://<server-host>:5801/eval_vln \
  --instruction "Walk forward and immediately stop when you exit the room." \
  --session-id go2-001
```

如果你把 3 个文件单独拷到 `realworld/` 目录并在该目录运行：

```bash
python go2_vln_client.py \
  --server-url http://<server-host>:5801/eval_vln \
  --instruction "Walk forward and immediately stop when you exit the room." \
  --session-id go2-001
```

可选参数：

- `--request-timeout 150`
- `--camera-topic <camera-topic>`：相机话题与当前 ROS2 配置不一致时使用
- `--odom-topic <odom-topic>`：切换到其他 `SportModeState` 话题时使用
- `--cmd-topic <command-topic>`：控制话题与当前 Unitree 配置不一致时使用

## GO2 端最小检查项

启动前建议检查：

1. 能 import `requests`、`numpy`、`PIL`
2. 能 import `rclpy`、`cv_bridge`
3. 能 import `unitree_go.msg`、`unitree_api.msg`
4. 相机话题 `/camera/camera/color/image_raw` 存在
5. 里程计话题 `/sportmodestate` 存在
6. GO2 能访问远程服务器 `http://<server-host>:5801/health`

可以手动测一下网络：

```bash
curl http://<server-host>:5801/health
```

## 远程 server 端和 GO2 端的依赖区别

### GO2 端需要

- `numpy`
- `requests`
- `Pillow`
- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `unitree_go`
- `unitree_api`

### GO2 端不需要

- `flask`
- `vllm`
- `transformers`
- `ms-swift`

### 远程 server 端需要

- `flask`
- `requests`
- `numpy`
- `Pillow`

以及远程 vLLM 服务本身需要的模型推理环境。

## 备注

如果 GO2 端环境里没有 `Pillow`，客户端无法把图像编码成 JPEG 上传。

如果 GO2 端环境里没有 `cv_bridge` 或 Unitree ROS2 消息包，客户端虽然能解析命令行，但无法实际订阅传感器和控制机器人。
