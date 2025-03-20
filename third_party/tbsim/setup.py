from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

# 直接设置描述，不读取README文件
long_description = "Traffic Behavior Simulation"

setup(
    name="tbsim",
    packages=[package for package in find_packages() if package.startswith("tbsim")],
    install_requires=[
        # "pymap3d",
        # "transforms3d",
        # "h5py",
        # "imageio",
        # "prettytable",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Traffic Behavior Simulation",
    author="NVIDIA AV Research",
    author_email="danfeix@nvidia.com",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
