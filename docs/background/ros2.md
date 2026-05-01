# ROS2 & Spot Telemetry

## What is ROS2?

ROS2 (Robot Operating System 2) is a middleware framework for building robot software. It provides a publish-subscribe messaging system that lets separate processes — running on the same machine or across a network — exchange sensor data, commands, and state without being tightly coupled.

Key concepts:

| Term | Meaning |
|------|---------|
| **Node** | A single process that participates in the ROS2 graph |
| **Topic** | A named channel over which nodes publish or subscribe to messages |
| **Message** | A typed data structure exchanged on a topic (e.g. `sensor_msgs/Image`) |
| **`rclpy`** | The Python client library for ROS2 |

In this project, ROS2 is used as the runtime bus that carries Spot's live sensor data and the computed scene graph to downstream consumers (navigation planners, visualisers, logging tools).

---

## Spot Telemetry Bridge

The file `spot_semantic_mapping/spot/telemetry.py` implements two publisher classes that bridge the Boston Dynamics Spot SDK to ROS2.

### Published Topics

| Topic | Message type | Description |
|-------|-------------|-------------|
| `/spot/pose` | `geometry_msgs/PoseStamped` | Robot body pose in the odometry frame |
| `/spot/graph` | `std_msgs/String` | Scene graph serialised as JSON |
| `/spot/twist` | `geometry_msgs/TwistStamped` | Robot linear and angular velocity |
| `/spot/battery` | `sensor_msgs/BatteryState` | Battery level and status |
| `/spot/cameras/<cam>/color` | `sensor_msgs/Image` | RGB image for each active camera |
| `/spot/cameras/<cam>/depth` | `sensor_msgs/Image` | Depth image for each active camera |

`<cam>` is the Spot camera source name (e.g. `frontleft`, `frontright`, `hand_color`).

### Publisher Classes

**`SpotObservationPublisher`** — async-based, ~5 Hz

Suitable for low-bandwidth consumers such as a scene graph updater or a map visualiser. Uses an asyncio event loop in a separate thread.

```python
from spot_semantic_mapping.spot.telemetry import main_standard
main_standard()  # blocks; press Ctrl+C to stop
```

**`FastSpotObservationPublisher`** — thread-safe, up to 50 Hz

Suitable for high-rate consumers such as a real-time navigation stack. Uses synchronous `agent.get_observation()` snapshots with a dedicated publisher thread.

```python
from spot_semantic_mapping.spot.telemetry import main_fast
main_fast()  # blocks; press Ctrl+C to stop
```

!!! note "Hostname"
    Both entry points currently have the lab Spot IP (`137.146.188.170`) hardcoded. Update the `hostname` argument in `telemetry.py` if you are connecting to a different robot.

---

## Typical Workflow with ROS2

```
Spot SDK  ──►  telemetry.py  ──►  ROS2 topics  ──►  your node
                                  /spot/pose
                                  /spot/graph
                                  /spot/cameras/…
```

1. Ensure ROS2 is sourced in your shell (`source /opt/ros/humble/setup.bash`).
2. Start the telemetry bridge (`main_standard` or `main_fast`).
3. Subscribe to the relevant topics in your consumer node using `rclpy`.

---

## Further Reading

- [ROS2 Humble documentation](https://docs.ros.org/en/humble/)
- [Boston Dynamics Spot SDK](https://dev.bostondynamics.com/)
- Implementation: `spot_semantic_mapping/spot/telemetry.py`
