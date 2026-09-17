# Per-press feedback. Published on /sim/press_feedback ONLY when the simulator
# is launched with debug_press_feedback:=true. Off by default: during a real
# attempt you learn the outcome from /sim/result alone.
std_msgs/Header header
bool    accepted
string  reason      # ACCEPTED, NO_INTERSECT, OUT_OF_REACH, TOO_CLOSE, GLANCING, NO_KEY, MOVING, DEBOUNCE
string  key         # key name if accepted, else empty
float64 board_x     # stylus/panel intersection in the board frame, if it exists
float64 board_y
float64 range       # metres along the aimed ray to the panel
float64 incidence   # radians from the panel normal
