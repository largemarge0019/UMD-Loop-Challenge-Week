# Published once on /sim/result after your node publishes /sim/done.
# This is the ONLY place the simulator reveals which keys registered.
std_msgs/Header header
int64   seed
string  target              # the launch key you were asked to type
string  typed               # what the simulator actually registered
bool    exact_match
uint32  edit_distance       # Levenshtein(target, typed)
uint32  presses_attempted
uint32  presses_accepted
string[] rejection_reasons  # parallel arrays: reason name -> count
uint32[] rejection_counts
float64 elapsed             # seconds from episode start to /sim/done
