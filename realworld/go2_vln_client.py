import argparse
import io
import json
import math
import queue
import threading
import time
from urllib.parse import urljoin
from uuid import uuid4

import numpy as np
import PIL.Image as PIL_Image
import rclpy
import requests
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# unitree related
from unitree_api.msg import Request
from unitree_api.msg import RequestHeader
from unitree_go.msg import SportModeState

try:
    from realworld.pid_controller import PID_controller
    from realworld.utils import ReadWriteLock
except ImportError:
    from pid_controller import PID_controller
    from utils import ReadWriteLock


pid = PID_controller(Kp_trans=1.0, Kd_trans=0.4, Kp_yaw=2.0, Kd_yaw=0.4, max_v=0.25, max_w=0.8)
manager = None

rgb_rw_lock = ReadWriteLock()
depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()


def eval_vln(image, instruction, session_id, reset, run_id, url, timeout):
    image = PIL_Image.fromarray(image[:, :, ::-1])
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    payload = {
        'reset': reset,
        'instruction': instruction,
        'session_id': session_id,
        'run_id': run_id,
    }
    files = {'image': ('rgb_image.jpg', image_bytes, 'image/jpeg')}

    start = time.time()
    try:
        response = requests.post(
            url,
            files=files,
            data={'json': json.dumps(payload)},
            timeout=timeout,
        )
        latency = time.time() - start
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        print(f'[eval_vln] request failed after {time.time() - start:.2f}s: {exc}')
        return [0]

    print(f'[eval_vln] total time(delay + policy): {latency:.2f}s')
    print(f'[eval_vln] response: {result}')

    if not result.get('success', True):
        print(f"[eval_vln] server returned error: {result.get('error', 'unknown error')}")
        return [0]

    actions = result.get('action', [0])
    if not isinstance(actions, list) or not actions:
        return [0]
    return actions


def image_to_jpeg_bytes(image):
    pil_image = PIL_Image.fromarray(image[:, :, ::-1])
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    return image_bytes


def log_frame(image, instruction, session_id, reset, run_id, frame_id, url, timeout):
    image_bytes = image_to_jpeg_bytes(image)
    payload = {
        'reset': reset,
        'instruction': instruction,
        'session_id': session_id,
        'run_id': run_id,
        'frame_id': frame_id,
        'timestamp': time.time(),
    }
    files = {'image': ('rgb_frame.jpg', image_bytes, 'image/jpeg')}

    response = requests.post(
        url,
        files=files,
        data={'json': json.dumps(payload)},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get('success', True):
        raise RuntimeError(result.get('error', 'unknown server error'))


def control_thread():
    while True:
        if manager.navigation_finished:
            manager.move(0.0, 0.0, 0.0)
            time.sleep(0.1)
            continue

        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            manager.move(v, 0, w)
        if abs(e_p) < 0.1 and abs(e_r) < 0.1:
            manager.trigger_replan()
        time.sleep(0.1)


def frame_logging_thread():
    while True:
        item = manager.frame_log_queue.get()
        if item is None:
            break

        frame_id, rgb_image, reset = item
        try:
            log_frame(
                image=rgb_image,
                instruction=manager.instruction,
                session_id=manager.session_id,
                reset=reset,
                run_id=manager.run_id,
                frame_id=frame_id,
                url=manager.frame_log_url,
                timeout=manager.frame_log_timeout,
            )
        except Exception as exc:
            print(f'[log_frame] frame_id={frame_id} failed: {exc}')
        finally:
            manager.frame_log_queue.task_done()


def planning_thread():
    while True:
        if not manager.should_plan:
            time.sleep(0.05)
            continue

        print('planning_thread running')
        rgb_rw_lock.acquire_read()
        rgb_image = None if manager.rgb_image is None else manager.rgb_image.copy()
        rgb_rw_lock.release_read()

        odom_rw_lock.acquire_read()
        request_cnt = manager.request_cnt
        odom_rw_lock.release_read()

        if rgb_image is None:
            time.sleep(0.1)
            continue

        odom_rw_lock.acquire_read()
        has_odom = manager.homo_odom is not None and manager.homo_goal is not None
        odom_rw_lock.release_read()
        if not has_odom:
            time.sleep(0.1)
            continue

        reset = manager.consume_reset_flag()
        actions = eval_vln(
            image=rgb_image,
            instruction=manager.instruction,
            session_id=manager.session_id,
            reset=reset,
            run_id=manager.run_id,
            url=manager.server_url,
            timeout=manager.request_timeout,
        )

        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.navigation_finished = 0 in actions
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()

        print(f'[planning_thread] request_cnt={request_cnt} actions={actions}')
        time.sleep(0.1)


class Go2VlnManager(Node):
    def __init__(
        self,
        instruction,
        server_url,
        session_id,
        request_timeout,
        frame_log_url,
        frame_log_timeout,
        frame_log_queue_size,
        camera_topic,
        odom_topic,
        cmd_topic,
    ):
        super().__init__('go2_manager')

        self.camera_topic = camera_topic
        self.odom_topic = odom_topic
        self.cmd_topic = cmd_topic
        self.rgb_sub = self.create_subscription(Image, self.camera_topic, self.rgb_callback, 1)
        self.odom_sub = self.create_subscription(SportModeState, self.odom_topic, self.odom_callback, 10)
        self.control_pub = self.create_publisher(Request, self.cmd_topic, 5)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None

        self.instruction = instruction
        self.server_url = server_url.rstrip('/')
        self.frame_log_url = frame_log_url
        self.session_id = session_id
        self.run_id = f'{session_id}-{uuid4().hex}'
        self.request_timeout = request_timeout
        self.frame_log_timeout = frame_log_timeout
        self.next_request_reset = True
        self.next_frame_log_reset = True

        self.request_cnt = 0
        self.frame_cnt = 0
        self.odom_cnt = 0
        self.should_plan = False
        self.navigation_finished = False
        self.last_plan_time = 0.0
        self.frame_log_queue = queue.Queue(maxsize=max(0, int(frame_log_queue_size)))

        self.get_logger().info(f'server_url={self.server_url}')
        self.get_logger().info(f'frame_log_url={self.frame_log_url}')
        queue_size = self.frame_log_queue.maxsize or 'unbounded'
        self.get_logger().info(f'frame_log_queue_size={queue_size}')
        self.get_logger().info(f'camera_topic={self.camera_topic}')
        self.get_logger().info(f'odom_topic={self.odom_topic}')
        self.get_logger().info(f'cmd_topic={self.cmd_topic}')
        self.get_logger().info(f'session_id={self.session_id}')
        self.get_logger().info(f'run_id={self.run_id}')
        self.get_logger().info(f'instruction={self.instruction}')

    def consume_reset_flag(self):
        reset = self.next_request_reset
        self.next_request_reset = False
        return reset

    def rgb_callback(self, msg):
        rgb_rw_lock.acquire_write()
        raw_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')[:, :, :]
        self.rgb_image = raw_image
        self.frame_cnt += 1
        frame_id = self.frame_cnt
        reset = self.next_frame_log_reset
        self.next_frame_log_reset = False
        rgb_rw_lock.release_write()
        log_item = (frame_id, raw_image.copy(), reset)
        try:
            self.frame_log_queue.put_nowait(log_item)
        except queue.Full:
            try:
                self.frame_log_queue.get_nowait()
                self.frame_log_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.frame_log_queue.put_nowait(log_item)
                self.get_logger().warn('[log_frame] queue full; dropped oldest pending frame')
            except queue.Full:
                self.get_logger().warn('[log_frame] queue full; dropped current frame')

    def depth_callback(self, msg):
        depth_rw_lock.acquire_write()
        if self.rgb_image is None:
            depth_rw_lock.release_write()
            return
        raw_depth = self.cv_bridge.imgmsg_to_cv2(msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth_rw_lock.release_write()

    def odom_callback(self, msg):
        downsample_ratio = 5
        self.odom_cnt += 1
        if self.odom_cnt % downsample_ratio != 0:
            return
        odom_rw_lock.acquire_write()
        rotation = np.array([
            [np.cos(msg.imu_state.rpy[2]), -np.sin(msg.imu_state.rpy[2])],
            [np.sin(msg.imu_state.rpy[2]), np.cos(msg.imu_state.rpy[2])],
        ])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = rotation
        self.homo_odom[:2, 3] = [msg.position[0], msg.position[1]]
        self.vel = [msg.velocity[0], msg.yaw_speed]

        if self.odom_cnt == downsample_ratio:
            self.homo_goal = self.homo_odom.copy()
        odom_rw_lock.release_write()

    def trigger_replan(self):
        self.should_plan = True

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError('Please initialize homo_goal before change it!')

        # A stop response must cancel the current target. Otherwise the
        # controller would continue moving toward the previous target after a
        # model failure or a completed episode.
        if 0 in actions:
            if self.homo_odom is not None:
                self.homo_goal = self.homo_odom.copy()
            return

        homo_goal = self.homo_goal

        for each_action in actions:
            if each_action == 0:
                continue
            if each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1],
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1],
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        sport_api_id_move = 1008
        payload = {'x': vx, 'y': vy, 'z': vyaw}
        parameter = json.dumps(payload)
        header = RequestHeader()
        header.identity.api_id = sport_api_id_move
        header.identity.id = time.monotonic_ns()
        request_msg = Request(parameter=parameter, header=header)
        self.control_pub.publish(request_msg)


def parse_args():
    parser = argparse.ArgumentParser(description='GO2 VLN client for the realworld vLLM server.')
    parser.add_argument(
        '--server-url',
        type=str,
        required=True,
        help='Realworld VLN server endpoint.',
    )
    parser.add_argument('--instruction', type=str, required=True, help='Global navigation instruction.')
    parser.add_argument('--session-id', type=str, default='go2', help='Session id used by the server.')
    parser.add_argument('--request-timeout', type=float, default=150.0, help='HTTP request timeout in seconds.')
    parser.add_argument(
        '--frame-log-url',
        type=str,
        default='',
        help='Frame logging endpoint. Defaults to /log_frame on the same server as --server-url.',
    )
    parser.add_argument('--frame-log-timeout', type=float, default=10.0, help='HTTP timeout for per-frame logging.')
    parser.add_argument(
        '--frame-log-queue-size',
        type=int,
        default=0,
        help='Maximum pending frames for asynchronous server-side frame logging. 0 means unbounded.',
    )
    parser.add_argument(
        '--camera-topic',
        type=str,
        default='/camera/camera/color/image_raw',
        help='ROS2 RGB camera topic.',
    )
    parser.add_argument(
        '--odom-topic',
        type=str,
        default='/sportmodestate',
        help='ROS2 Unitree SportModeState topic.',
    )
    parser.add_argument(
        '--cmd-topic',
        type=str,
        default='/api/sport/request',
        help='ROS2 Unitree sport request topic.',
    )
    return parser.parse_args()


def default_frame_log_url(server_url):
    return urljoin(server_url.rstrip('/') + '/', '../log_frame')


if __name__ == '__main__':
    args = parse_args()
    frame_log_url = args.frame_log_url or default_frame_log_url(args.server_url)

    control_thread_instance = threading.Thread(target=control_thread, daemon=True)
    planning_thread_instance = threading.Thread(target=planning_thread, daemon=True)
    frame_logging_thread_instance = threading.Thread(target=frame_logging_thread, daemon=True)

    rclpy.init()

    try:
        manager = Go2VlnManager(
            instruction=args.instruction,
            server_url=args.server_url,
            session_id=args.session_id,
            request_timeout=args.request_timeout,
            frame_log_url=frame_log_url,
            frame_log_timeout=args.frame_log_timeout,
            frame_log_queue_size=args.frame_log_queue_size,
            camera_topic=args.camera_topic,
            odom_topic=args.odom_topic,
            cmd_topic=args.cmd_topic,
        )

        control_thread_instance.start()
        planning_thread_instance.start()
        frame_logging_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.destroy_node()
        rclpy.shutdown()
