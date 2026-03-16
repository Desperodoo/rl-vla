VLAW (arXiv:2602.12063) Action Space Analysis
1. Overall Structure

VLAW does not introduce a new action space.
Instead, it inherits the action interface from the pretrained VLA policy 
𝜋
0.5
π
0.5
	​

 and the Ctrl-World world model.

The system consists of two main components:

Policy (π0.5 / DROID)
        ↓
Action Chunk (low-level robot command)
        ↓
Robot Execution
        ↓
Adapter + Forward Kinematics
        ↓
Cartesian Pose Sequence
        ↓
World Model (Ctrl-World)

Key idea:

Policy operates in joint-space

World model conditions on Cartesian pose space

These two spaces are connected through adapter + forward kinematics (FK).

2. Policy Action Space

The policy action space is inherited from the pretrained VLA policy 
𝜋
0.5
π
0.5
	​

, which itself uses the DROID policy interface.

Action Type
joint-space low-level control

Specifically:

joint velocity commands
Action Structure

The policy predicts action chunks instead of single-step commands.

Example:

a_t = [a_t^1, a_t^2, ..., a_t^H]

Typical horizon:

H = 15 steps  (Ctrl-World setup)
H ≈ 50 steps (~1 second in π0.5)

Each element in the chunk represents:

joint velocity vector

Example (Franka arm):

a_t^i ∈ R^7
Interpretation

The policy outputs:

future joint velocity trajectory

which the robot executes open-loop for the chunk duration.

3. World Model Action Conditioning

The world model does not condition directly on joint actions.

Instead, actions are transformed into Cartesian-space pose trajectories.

Pipeline:

joint velocities
      ↓
adapter
      ↓
future joint configuration
      ↓
forward kinematics (FK)
      ↓
Cartesian arm pose sequence
      ↓
world model condition

Thus the world model action condition space is:

Cartesian robot arm pose sequence

This typically includes:

end-effector pose
or
full arm pose representation
4. Why Convert Actions to Cartesian Pose

The main motivation is:

Geometric consistency for video prediction

The world model predicts future camera frames.

Robot motion in images correlates more directly with:

Cartesian arm pose

than with:

joint velocities

Thus using pose-space conditioning makes the visual prediction task easier.

5. Resulting Dual Action Representation

The system therefore uses two different action representations:

Component	Action Representation
Policy	joint velocity chunk
World Model	Cartesian pose sequence

Relationship:

joint velocities
        ↓
integration
        ↓
joint trajectory
        ↓
forward kinematics
        ↓
Cartesian pose trajectory
6. Relation to Common Robotics Action Spaces

Mapping to common robotics terminology:

Action Space	Used in VLAW
joint velocity	✔ policy action
joint position	✘
ee_pose	✔ world model condition (via FK)
ee_delta_pose	✘

Thus VLAW combines:

joint-space control policy
+
Cartesian-space world model
7. Key Design Insight

The architecture separates control interface and world modeling interface.

Policy:

learns robot control

World model:

learns visual future conditioned on geometry

This separation allows:

reuse of existing robot policies

more stable world model training

better geometric grounding for video prediction

8. Final Summary

Policy Action Space

joint velocity action chunk

World Model Action Conditioning

future Cartesian robot pose sequence

Bridge between them:

joint velocity
    → adapter
    → joint trajectory
    → forward kinematics
    → Cartesian pose trajectory

Thus:

policy space ≠ world model space

The two are connected through robot kinematics.