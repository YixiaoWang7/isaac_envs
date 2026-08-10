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

The generated catalog has 36 semantic tasks across tea, candy, combined service, and takeaway families. Each semantic
task has two prompt variants. Candy prompts deliberately leave the serving container unspecified; mug and bowl are
both valid, but all requested candies must use the same selected container. Named `orthogonal_train`, `id_eval`, and
`ood_recombination` splits are available through the task command configuration.

State and vision environment IDs:

```text
Isaac-CG-DeskService-Franka-IK-Rel-v0
Isaac-CG-DeskService-Franka-IK-Rel-Vision-v0
```

## Teleoperation and recording

SpaceMouse provides six relative end-effector axes plus its left-button gripper toggle. Its right button resets the
episode. Keyboard is available as a fallback.

```bash
python scripts/teleop.py --teleop-device spacemouse
python scripts/teleop.py --teleop-device keyboard --task-id 0
python scripts/record_demos.py --enable-cameras --teleop-device spacemouse \
  --dataset-file datasets/desk_service.hdf5 --num-demos 20
python scripts/inspect_dataset.py datasets/desk_service.hdf5
```

Only demonstrations satisfying the complete goal for ten consecutive steps are exported. Datasets include simulator
state, policy observations, raw and processed actions, task ID, prompt variant/tokens, factor encoding, stage progress,
seed, and catalog metadata.

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
