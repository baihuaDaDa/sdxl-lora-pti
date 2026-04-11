import os

from setuptools import find_packages, setup


def read_requirements(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="lora_diffusion",
    py_modules=["lora_diffusion"],
    version="0.1.7",
    description="Low Rank Adaptation for Diffusion Models. Works with Stable Diffusion out-of-the-box.",
    author="Simo Ryu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lora_add = lora_diffusion.cli_lora_add:main",
            "lora_pti = lora_diffusion.cli_lora_pti:main",
            "lora_distill = lora_diffusion.cli_svd:main",
            "lora_ppim = lora_diffusion.preprocess_files:main",
        ],
    },
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
)
