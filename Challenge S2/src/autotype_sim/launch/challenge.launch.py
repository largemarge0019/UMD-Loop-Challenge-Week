"""Launch the URC autonomous-typing simulator.

One node: physics, rendering, press evaluation AND the browser dashboard. The
dashboard is embedded in the sim node on purpose — it displays operator
information (board pose, which key registered) that does not appear on any
ROS topic a member's node could subscribe to.

    ros2 launch autotype_sim challenge.launch.py seed:=3 launch_key:=MARS

Arguments
  seed                 Board-pose seed. Same seed → same board pose. (int)
  launch_key           3–6 characters, A–Z / 0–9. The sim publishes it on
                       /sim/launch_key and scores against it. Passed to the
                       node as a string, so keys that YAML would read as
                       booleans (YES, OFF, TRUE …) arrive unchanged instead of
                       being silently rewritten. A key spelled like a number in
                       rcl's own parameter YAML (1E5, 0X1F) still cannot
                       survive the round trip — the node refuses to start and
                       says so; generate a different key.
  teleop               Enable joint-jog controls and a press button in the
                       dashboard. For checking the sim, not for attempts.
  debug_press_feedback Publish /sim/press_feedback after every press. Off for
                       real attempts — the outcome is only in /sim/result.
  port                 Dashboard HTTP/WebSocket port.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    seed = LaunchConfiguration("seed")
    # value_type=str: without it launch_ros type-infers the substitution with
    # YAML rules, so a legal 3-6 char A-Z/0-9 key that happens to look like a
    # YAML scalar is silently changed (YES -> True -> "TRUE", OFF -> "FALSE")
    # or rejected (1E5 -> 100000.0, 0X1F -> 31).
    launch_key = ParameterValue(LaunchConfiguration("launch_key"), value_type=str)
    teleop = LaunchConfiguration("teleop")
    debug_press_feedback = LaunchConfiguration("debug_press_feedback")
    port = LaunchConfiguration("port")

    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="1"),
            DeclareLaunchArgument("launch_key", default_value="ROVER"),
            DeclareLaunchArgument("teleop", default_value="false"),
            DeclareLaunchArgument("debug_press_feedback", default_value="false"),
            DeclareLaunchArgument("port", default_value="8080"),
            Node(
                package="autotype_sim",
                executable="sim_node",
                name="autotype_sim",
                output="screen",
                parameters=[
                    {
                        "seed": seed,
                        "launch_key": launch_key,
                        "teleop": teleop,
                        "debug_press_feedback": debug_press_feedback,
                        "dashboard_port": port,
                    }
                ],
            ),
        ]
    )
