std_msgs/Header header
int32 drone_id

uint8 state
float64 state_time_s
uint32 state_transitions

uint32 plan_attempts
uint32 plan_successes
uint32 plan_failures
uint32 plan_no_grid

uint32 astar_failures
uint32 kino_search_failures
uint32 kino_optimize_failures
uint32 yaw_failures
uint32 grid_tsp_failures
uint32 frontier_tsp_failures
uint32 refine_failures

uint32 collision_static
uint32 collision_swarm
float64 last_collision_distance
int32 last_collision_other_id

uint32 pair_opt_requests
uint32 pair_opt_accepted
uint32 pair_opt_rejected
uint32 pair_opt_timeouts
uint32 pair_opt_optimize_failures
uint32 pair_opt_no_candidate
uint32 pair_opt_no_improvement

uint32 trajectory_published
uint32 trajectory_dropped_old
uint32 trajectory_completed

string[] stage_names
uint32[] stage_counts
float64[] stage_last_ms
float64[] stage_avg_ms
float64[] stage_max_ms

string[] state_names
float64[] state_times_s
uint32[] state_entries

string[] gauge_names
float64[] gauge_values
