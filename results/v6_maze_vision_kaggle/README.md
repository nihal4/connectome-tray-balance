# v6: fly-robot brain on the camera maze (`maze_fixed3_vision`, Kaggle, one seed) — in progress

The fly-robot brain (2,500 neurons, eye to command neurons) sees the maze only through the
16x16 camera. First run: 0 % solved, eval return rose from −9.5 to about +5.3 by 1.05M steps
and then stayed flat.

**What it actually does** (`pinned_in_goal_corner.mp4`): it rolls the ball through the whole
maze to the goal in under a second, then pins it in the corner of the goal cell against the
walls, 4.8 cm from the goal centre. Solving needs the ball to rest within 3 cm of the centre
for 0.5 s. A ball pinned in that corner gives exactly the observed return (~5.5), so the
navigation is learned and only the final "stop in the middle" is missing.

Why: pinning against walls is stable and easy; balancing in the open middle of the cell is a
fine-control skill, and the reward for the centre versus the corner was under 1 point.

**Changes after this run:**
1. Start-position curriculum (`--random_start 0.5`).
2. A stronger, potential-based pull toward the goal centre inside the goal cell
   (`GOAL_PULL`, worth about +10 for moving from the corner to the centre; given back if the
   ball leaves). The success rule is unchanged. On the new reward scale a pinned ball scores
   about 6 and a full solve about 39.
