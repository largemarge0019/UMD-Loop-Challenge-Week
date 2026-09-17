# Velocity command for the 5-joint typing arm. Joints are matched BY NAME, so
# order does not matter, but every name must be one of the five published on
# /joint_states. Units: rad/s. Any joint you omit keeps its last command.
#
# The simulator zeroes all velocities if no command arrives within the watchdog
# window (see INTERFACES.md), so this must be published continuously from a loop.
std_msgs/Header header
string[] name
float64[] velocity
