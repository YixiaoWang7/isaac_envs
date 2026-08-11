# CG Isaac Lab Desk-Service Environment

A standalone Isaac Lab extension for prompt-conditioned, long-horizon manipulation and compositional-generalization
experiments. All task objects remain on the desk in every episode; the prompt determines which nested relations must
be achieved. Goals are final-state graphs, so valid action orderings are not hard-coded.

## Installation

For a clean-machine setup—including the pinned Python, PyTorch, Isaac Sim, and Isaac Lab installation—follow
[`SETUP.md`](SETUP.md). For an already configured Isaac Lab environment, install this extension with:

```bash
source /path/to/env_isaaclab/bin/activate
cd /path/to/isaac_envs
python -m pip install -e source/cg_isaac_envs
```

Isaac Sim requires one-time interactive acceptance of NVIDIA's EULA on first launch. The scripts do not accept it on
the user's behalf.

List the registered environments:

```bash
python scripts/list_envs.py
```

## Tasks

The household catalog contains `3 × 3 × 3 = 27` tasks. A task selects one colored cube, one mug, and one destination
station. Each task has two prompt variants, while successful-demo quotas are balanced by task ID.

State and vision environment IDs:

```text
Isaac-CG-Household-Franka-IK-Rel-v0
Isaac-CG-Household-Franka-IK-Rel-Vision-v0
```

## Teleoperation and recording

### Prepare the terminal

Teleoperation is graphical. Run it from the logged-in desktop session and do not pass `--headless`. Activate the
installed IsaacLab environment and use a writable temporary directory:

```bash
cd ~/code/cg/isaac_envs
source ../env_isaaclab/bin/activate
mkdir -p ../.isaaclab_tmp
export TMPDIR="$(realpath ../.isaaclab_tmp)"
```

The custom extension must already be installed in this environment:

```bash
python -m pip install -e source/cg_isaac_envs
```

### SpaceMouse setup

The default `hid` backend uses `pyspacemouse` and automatically searches the readable `/dev/hidraw*` interfaces:

```bash
python scripts/teleop.py --teleop-device spacemouse
```

The tested Linux input-event backend can instead use the known event node explicitly:

```bash
python scripts/teleop.py \
  --teleop-device spacemouse \
  --spacemouse-backend input \
  --spacemouse-device /dev/input/event4
```

Input-event numbers can change after reconnecting the device or rebooting. Check the current nodes and stable symlinks
before launching:

```bash
ls -l /dev/input/event* /dev/input/by-id/*3Dconnexion* 2>/dev/null
```

The selected device must be readable by the current user. If opening it reports `Permission denied`, configure a
device-specific udev rule or add the user to the machine's input-device access group, then log out and back in. Do not
run Isaac Sim permanently with `sudo`.

SpaceMouse controls:

- Translation of the cap commands relative end-effector translation.
- Twist commands yaw. Roll and pitch remain locked to the straight-down gripper orientation.
- Button 1 toggles the persistent gripper target between open and closed.
- During data collection, a reset task first enters `READY` mode so the prompt and scene can be inspected. Button 2
  starts recording from `READY`. While `RECORDING`, button 2 discards the entire attempt, resets the same task, and
  returns to `READY`; press it again when prepared to start the replacement attempt.
- After completing the placement, open the gripper and keep the released cube and mug stationary for one second. The
  attempt is saved only after this full verification period.
- `--sensitivity VALUE` scales translation and rotation speed; the default is `1.0`.

The main viewport shows the front camera. Separate wrist and robot-right side windows open at the same time. The active
task prompt is printed in the terminal; the recording script also shows it in an Isaac UI progress panel.

To launch a fixed task, use a task ID from `0` through `26`:

```bash
python scripts/teleop.py --teleop-device spacemouse --task-id 0
```

Keyboard is available as a fallback:

```bash
python scripts/teleop.py --teleop-device keyboard --task-id 0
```

### Balanced demonstration collection

Supply both the dataset directory and the desired number of successful demonstrations per task. The recommended HID
backend uses the same SpaceMouse setup as interactive teleoperation:

```bash
python scripts/record_demos.py \
  --teleop-device spacemouse \
  --spacemouse-backend hid \
  --dataset-dir datasets/household_20_each \
  --demos-per-task 20 \
  --seed 1001
```

To collect only the `OA(9, 3³)` orthogonal nine-task design, add `--task-set orthogonal`:

| Run | Stage A | Stage B | Stage C | Task ID |
| --: | :-----: | :-----: | :-----: | ------: |
| 1 | 1 | 1 | 1 | 0 |
| 2 | 1 | 2 | 2 | 4 |
| 3 | 1 | 3 | 3 | 8 |
| 4 | 2 | 1 | 2 | 10 |
| 5 | 2 | 2 | 3 | 14 |
| 6 | 2 | 3 | 1 | 15 |
| 7 | 3 | 1 | 3 | 20 |
| 8 | 3 | 2 | 1 | 21 |
| 9 | 3 | 3 | 2 | 25 |

```bash
python scripts/record_demos.py \
  --teleop-device spacemouse \
  --spacemouse-backend hid \
  --task-set orthogonal \
  --dataset-dir datasets/household_orthogonal_10_each \
  --demos-per-task 10 \
  --seed 1001
```

For an arbitrary subset, use stage codes directly, for example
`--task-codes 111 122 133`. A dataset records its exact selected task IDs, so resume it with the same `--task-set` or
`--task-codes` selection and order.

If raw HID is unavailable but the Linux event interface has been verified, replace `--spacemouse-backend hid` with
`--spacemouse-backend input --spacemouse-device /dev/input/event4` (using the current event node).

The collector completes the requested quota for the current selected task before showing the next selected task, and
continues in the requested task-code order. A success requires the selected cube to be inside the target mug, the target
mug to be on its station, and the end-effector tool-tip to be at least 0.15 m above the target mug's center. These three
conditions must remain true for one continuous second (30 control steps by default); if any becomes false, the timer
restarts. The collection panel reports what is still missing.

Each HDF5 timestep contains the robot-base end-effector pose, measured gripper width, processed six-axis IK delta,
absolute binary gripper target, and post-action cube-in-mug, mug-on-station, end-effector-height, and held-success flags.
Three synchronized 256 px MP4 files contain the front, wrist, and side views. The collector enforces equal
successful-episode quotas across task IDs and safely resumes an incomplete compatible dataset. Its gripper action is a
step-function target (`0 = closed`, `1 = open`), not a delta. The verification duration can be changed with
`--success-hold-seconds`, although one second is recommended.

Use a different `--seed` with a different dataset directory to collect another independent layout sequence. Reuse the
original seed when resuming an existing dataset; the collector rejects a mismatched seed to prevent accidental mixing.
The exact task subset and success definition are also part of the dataset schema. If either changes, start a new
dataset directory rather than mixing demonstrations collected under different criteria.

Successful episodes are stored in `dataset.hdf5`, with three corresponding MP4 files under `videos/`. Operator-reset,
dropped-object, and interrupted attempts are removed rather than added to the dataset. Inspect a collection with:

```bash
python scripts/inspect_dataset.py datasets/household_20_each/dataset.hdf5
```

## Evaluation

The evaluator supports zero, random, privileged scripted, and external policies. External factories use
`module:function` and receive `env`, `checkpoint`, and `device`; returned policies implement `reset(env_ids, tasks)` and
`act(observations, tasks) -> [num_envs, 7]`.

```bash
python scripts/evaluate.py --headless --policy scripted --num-envs 64 --episodes-per-task 5
python scripts/evaluate.py --headless --policy my_policy:create --checkpoint checkpoint.pt \
  --task-split ood_recombination
```

Results are written as per-episode CSV plus JSON/CSV summaries covering task, family, destination, stages, and failure
categories. The scripted controller uses privileged simulator poses and is intended for plumbing/physics validation,
not as a language-policy result.

## Assets and tests

Core physics uses local procedural assets. Optional NVIDIA visual USD layers can be localized after EULA acceptance:

```bash
python scripts/prepare_assets.py --headless
```

Run pure-Python tests without launching Isaac Sim:

```bash
pytest -q
```

Run the simulator smoke test after the first-launch prerequisite:

```bash
python tests/smoke_env.py --headless --num-envs 1
python tests/smoke_env.py --headless --num-envs 64
```
