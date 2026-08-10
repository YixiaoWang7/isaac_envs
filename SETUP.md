# Environment setup

This guide installs the complete `isaac_envs` development environment on a new Linux machine. The pinned versions
below reproduce the stack on which this repository was developed and tested.

## Tested stack

- Ubuntu 24.04 LTS
- NVIDIA driver 580.173.02 (an RTX 5080 was used for development)
- Python 3.11
- PyTorch 2.7.0 with CUDA 12.8 wheels
- Isaac Sim 5.1.0
- Isaac Lab commit `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`

Isaac Sim is GPU-intensive. Install a current NVIDIA driver suitable for your GPU before continuing, and confirm that
`nvidia-smi` works. The exact GPU and driver above are a known-good configuration, not strict requirements.

## 1. Install system prerequisites

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3.11 python3.11-venv
```

If Python 3.11 is unavailable from your distribution packages, install Miniconda or `uv` and create a Python 3.11
environment with it. Isaac Sim 5.x requires Python 3.11; do not use the Ubuntu 24.04 default Python 3.12.

## 2. Create the workspace and virtual environment

The directory names may be changed. These commands keep Isaac Lab, this repository, and the virtual environment next
to one another, matching the development layout.

```bash
mkdir -p ~/code/cg
cd ~/code/cg
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate
python -m pip install --upgrade pip
```

Activate this environment again in every new shell before using the project:

```bash
source ~/code/cg/env_isaaclab/bin/activate
```

## 3. Install PyTorch and Isaac Sim

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

NVIDIA presents its Isaac Sim license/EULA on first launch. Review and accept it interactively; this repository does
not accept it automatically.

Verify the package before installing the remaining code:

```bash
python -m pip show isaacsim isaacsim-app
```

## 4. Clone and install the pinned Isaac Lab revision

```bash
cd ~/code/cg
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
./isaaclab.sh --install
```

`--install` installs all optional learning frameworks. They are not required for the basic environment and smoke
tests; to make the installation smaller, use `./isaaclab.sh --install none` instead.

Confirm that the official Isaac Lab environment is registered:

```bash
python scripts/environments/list_envs.py --keyword Isaac-Lift-Cube-Franka-IK-Rel-v0
```

## 5. Clone and install this repository

```bash
cd ~/code/cg
git clone git@github.com:YixiaoWang7/isaac_envs.git
cd isaac_envs
python -m pip install -e source/cg_isaac_envs
```

If the machine does not have a GitHub SSH key configured, use:

```bash
git clone https://github.com/YixiaoWang7/isaac_envs.git
```

## 6. Verify the project

```bash
cd ~/code/cg/isaac_envs
pytest -q
python scripts/list_envs.py
python tests/smoke_env.py --headless --num-envs 1
```

After that succeeds, an optional parallel smoke test is:

```bash
python tests/smoke_env.py --headless --num-envs 64
```

## Assets, datasets, and generated output

Core physics uses the local assets committed to this repository. To fetch optional NVIDIA visual assets after
accepting the NVIDIA license, run:

```bash
python scripts/prepare_assets.py --headless
```

The following local data is intentionally excluded from Git:

- `outputs/`: videos, previews, evaluation summaries, and other generated results
- `datasets/`: recorded demonstrations
- `logs/`: simulator and training logs
- `assets/downloaded/`: downloaded asset payloads (the placeholder file is retained)

Copy any datasets or generated results separately when migrating machines. Do not force-add them to Git.

## Troubleshooting

- **Wrong Python version:** `python --version` must report Python 3.11 for Isaac Sim 5.1.
- **Isaac Sim startup failure:** confirm the environment is active, `nvidia-smi` works, and the driver supports the
  installed CUDA/PyTorch stack.
- **EULA or first-launch prompt:** launch once from an interactive terminal and complete NVIDIA's prompt.
- **Project environments are missing:** reinstall with `python -m pip install -e source/cg_isaac_envs`.
- **Headless server:** camera-based tasks still require a working NVIDIA rendering stack with `--headless`.

Record the exact installed versions when diagnosing another machine:

```bash
python --version
python -m pip show torch torchvision isaacsim isaaclab isaaclab-tasks
nvidia-smi
```
