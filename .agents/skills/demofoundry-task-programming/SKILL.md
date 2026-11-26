---
name: demofoundry-task-programming
description: Reconstruct a calibrated robot workcell, execute a task in simulation, and export synchronized proxy videos, states and actions. Use for the Marvin M6S example or to adapt it to another workcell.
---

## Run the Marvin example

Follow the [installation and input guide](../../../examples/marvin/README.md). Run `python examples/marvin/run.py --out runs/marvin` from the repository root. The [task program](../../../examples/marvin/src/program.py) places red on green with the left arm, then blue on red with the right.

## Adapt a workcell

1. Collect synchronized camera images, the initial joint and gripper positions, robot and gripper meshes, and object dimensions or RGB-D. Record intrinsics and lens distortion at the capture resolution. Supply the base registration for fixed cameras and the optical-to-flange transform for each wrist camera.
2. Fit object poses and support surfaces in the robot base frame. Check each simulated first frame against the real view. For another robot, adapt kinematics, collision geometry, visual meshes, tool frames and the controller’s joint order, units and rate.
3. Translate the instruction into object-relative motions and a measurable success condition. Execute the program with contact dynamics, release the objects and check their settled poses. For a new Marvin layout, pass the calibrated directory with `--workcell` and choose a new output directory.

## Return the demonstration

Inspect `fit/*_comparison.jpg` and `validation.json`. Return the proxy videos and `episode.npz`; its `frame_state_index` maps each video frame to the state measured before its action. For the rendered wrist clip, run `bash examples/marvin/render_wrist.sh runs/marvin` in the [H3 environment](../../../inference/h3/README.md#install).
