"""Editable installation for the CG Isaac environments."""

from setuptools import find_packages, setup

setup(
    name="cg-isaac-envs",
    version="0.1.0",
    description="Compositional generalization environments for Isaac Lab",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=["gymnasium", "h5py", "numpy", "opencv-python", "pyspacemouse==2.0.0", "torch"],
    zip_safe=False,
)
