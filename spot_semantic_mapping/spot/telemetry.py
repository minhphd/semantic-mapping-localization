"""
ROS2 telemetry bridge for Spot robot.

Provides two publisher nodes:
- SpotObservationPublisher: async-based, low-rate publisher
- FastSpotObservationPublisher: thread-safe, high-rate publisher (up to 50 Hz)
"""

import threading
import asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, BatteryState
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image as RosImage

from spot_semantic_mapping.spot.agent import SpotAgent
from spot_semantic_mapping.configs.loader import cfg


# ============================================================
# Utilities
# ============================================================

def np_to_ros_image(img_np, frame_id, stamp, encoding="bgr8"):
    msg = RosImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(img_np.shape[0])
    msg.width = int(img_np.shape[1])

    if encoding == "bgr8":
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(msg.width * 3)
        msg.data = img_np.astype(np.uint8).tobytes()
    elif encoding == "mono8":
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = int(msg.width)
        msg.data = img_np.astype(np.uint8).tobytes()
    elif encoding == "16UC1":
        msg.encoding = "16UC1"
        msg.is_bigendian = False
        msg.step = int(msg.width * 2)
        msg.data = img_np.astype(np.uint16).tobytes()
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return msg


def now_header(node: Node, frame_id: str) -> Header:
    h = Header()
    h.stamp = node.get_clock().now().to_msg()
    h.frame_id = frame_id
    return h


# ============================================================
# Standard publisher (async fetch)
# ============================================================

class SpotObservationPublisher(Node):
    """Async-based Spot observation publisher (~5 Hz)."""

    def __init__(self, spot_agent, publish_rate_hz=5.0):
        super().__init__("spot_observation_publisher")
        self.agent = spot_agent

        self.pub_pose = self.create_publisher(PoseStamped, "/spot/pose", 10)
        self.pub_graph = self.create_publisher(String, "/spot/graph", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/spot/twist", 10)
        self.pub_batt = self.create_publisher(BatteryState, "/spot/battery", 10)
        self.pub_img = {}

        self.period = 1.0 / float(publish_rate_hz)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._last_obs = None
        self._future = None
        self._schedule_fetch()
        self.timer = self.create_timer(self.period, self._tick)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def destroy_node(self):
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        super().destroy_node()

    def _get_img_pub(self, topic: str):
        if topic not in self.pub_img:
            self.pub_img[topic] = self.create_publisher(Image, topic, 10)
        return self.pub_img[topic]

    def _schedule_fetch(self):
        if self._future is None or self._future.done():
            self._future = asyncio.run_coroutine_threadsafe(
                self.agent.get_observation_async(), self._loop
            )

    def _tick(self):
        if self._future is not None and self._future.done():
            try:
                self._last_obs = self._future.result()
            except Exception as e:
                self.get_logger().error(f"Failed to get observation: {e}")
                self._last_obs = None
            finally:
                self._future = None

        self._schedule_fetch()

        obs = self._last_obs
        if obs is None:
            return

        self._publish_obs(obs)

    def _publish_obs(self, obs):
        stamp = self.get_clock().now().to_msg()

        b = BatteryState()
        b.header = now_header(self, "spot_body")
        b.percentage = float(obs.get("battery", 0.0)) / 100.0
        self.pub_batt.publish(b)

        twist = TwistStamped()
        twist.header = now_header(self, "spot_body")
        av = obs["imu"]["angular_velocity"]
        lv = obs["imu"]["linear_velocity"]
        twist.twist.angular.x = float(av["x"])
        twist.twist.angular.y = float(av["y"])
        twist.twist.angular.z = float(av["z"])
        twist.twist.linear.x = float(lv["x"])
        twist.twist.linear.y = float(lv["y"])
        twist.twist.linear.z = float(lv["z"])
        self.pub_twist.publish(twist)

        pose = PoseStamped()
        pose.header = now_header(self, "vision")
        p = obs["pose"]["position"]
        q = obs["pose"]["rotation"]
        pose.pose.position.x = float(p["x"])
        pose.pose.position.y = float(p["y"])
        pose.pose.position.z = float(p["z"])
        pose.pose.orientation.w = float(q["w"])
        pose.pose.orientation.x = float(q["x"])
        pose.pose.orientation.y = float(q["y"])
        pose.pose.orientation.z = float(q["z"])
        self.pub_pose.publish(pose)

        if 'g_text_desc' in obs:
            msg = String()
            msg.data = obs['g_text_desc']
            self.pub_graph.publish(msg)

        for cam_name, cam_data in obs.get("cameras", {}).items():
            if cam_data.get("color") is not None and isinstance(cam_data["color"], np.ndarray):
                msg = np_to_ros_image(cam_data["color"], f"{cam_name}_color", stamp, encoding="bgr8")
                msg.header = now_header(self, f"{cam_name}_color")
                self._get_img_pub(f"/spot/cameras/{cam_name}/color").publish(msg)

            if cam_data.get("depth") is not None and isinstance(cam_data["depth"], np.ndarray):
                msg = np_to_ros_image(cam_data["depth"], f"{cam_name}_depth", stamp, encoding="16UC1")
                msg.header = now_header(self, f"{cam_name}_depth")
                self._get_img_pub(f"/spot/cameras/{cam_name}/depth").publish(msg)


# ============================================================
# Fast publisher (thread-safe snapshot)
# ============================================================

class FastSpotObservationPublisher(Node):
    """
    High-rate Spot observation publisher (up to 50 Hz).
    Uses a thread-safe get_observation() snapshot instead of async fetches.
    """

    def __init__(self, spot_agent, publish_rate_hz=50.0):
        super().__init__("spot_observation_publisher")
        self.agent = spot_agent

        self.pub_pose = self.create_publisher(PoseStamped, "/spot/pose", 10)
        self.pub_graph = self.create_publisher(String, "/spot/graph", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/spot/twist", 10)
        self.pub_batt = self.create_publisher(BatteryState, "/spot/battery", 10)

        self.pub_img = {}
        for cam_name in self.agent.cameras.keys():
            self.pub_img[f"{cam_name}_color"] = self.create_publisher(
                Image, f"/spot/cameras/{cam_name}/color", 10
            )
            self.pub_img[f"{cam_name}_depth"] = self.create_publisher(
                Image, f"/spot/cameras/{cam_name}/depth", 10
            )

        self.period = 1.0 / float(publish_rate_hz)
        self.timer = self.create_timer(self.period, self._tick)

    def _tick(self):
        obs = self.agent.get_observation()
        if obs is None or not obs.get("cameras"):
            return

        sync_stamp = self.get_clock().now().to_msg()

        def make_header(frame_id):
            h = Header()
            h.stamp = sync_stamp
            h.frame_id = frame_id
            return h

        if obs.get("battery") is not None:
            b = BatteryState()
            b.header = make_header("spot_body")
            b.percentage = float(obs["battery"]) / 100.0
            self.pub_batt.publish(b)

        if obs.get("imu") and obs["imu"].get("angular_velocity"):
            twist = TwistStamped()
            twist.header = make_header("spot_body")
            av = obs["imu"]["angular_velocity"]
            lv = obs["imu"]["linear_velocity"]
            twist.twist.angular.x = float(av["x"])
            twist.twist.angular.y = float(av["y"])
            twist.twist.angular.z = float(av["z"])
            twist.twist.linear.x = float(lv["x"])
            twist.twist.linear.y = float(lv["y"])
            twist.twist.linear.z = float(lv["z"])
            self.pub_twist.publish(twist)

        if obs.get("pose") and obs["pose"].get("position"):
            pose = PoseStamped()
            pose.header = make_header("vision")
            p = obs["pose"]["position"]
            q = obs["pose"]["rotation"]
            pose.pose.position.x = float(p["x"])
            pose.pose.position.y = float(p["y"])
            pose.pose.position.z = float(p["z"])
            pose.pose.orientation.w = float(q["w"])
            pose.pose.orientation.x = float(q["x"])
            pose.pose.orientation.y = float(q["y"])
            pose.pose.orientation.z = float(q["z"])
            self.pub_pose.publish(pose)

        if obs.get('g_text_desc'):
            msg = String()
            msg.data = obs['g_text_desc']
            self.pub_graph.publish(msg)

        for cam_name, cam_data in obs.get("cameras", {}).items():
            if cam_data.get("color") is not None and isinstance(cam_data["color"], np.ndarray):
                msg = np_to_ros_image(cam_data["color"], f"{cam_name}_color", sync_stamp, encoding="bgr8")
                msg.header = make_header(f"{cam_name}_color")
                key = f"{cam_name}_color"
                if key in self.pub_img:
                    self.pub_img[key].publish(msg)

            if cam_data.get("depth") is not None and isinstance(cam_data["depth"], np.ndarray):
                msg = np_to_ros_image(cam_data["depth"], f"{cam_name}_depth", sync_stamp, encoding="16UC1")
                msg.header = make_header(f"{cam_name}_depth")
                key = f"{cam_name}_depth"
                if key in self.pub_img:
                    self.pub_img[key].publish(msg)


# ============================================================
# Entry points
# ============================================================

def main_standard():
    rclpy.init()
    agent = SpotAgent(hostname="137.146.188.170")
    node = SpotObservationPublisher(agent, publish_rate_hz=5.0)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main_fast():
    rclpy.init()
    agent = SpotAgent(hostname="137.146.188.170")
    node = FastSpotObservationPublisher(agent, publish_rate_hz=50.0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main_fast()
