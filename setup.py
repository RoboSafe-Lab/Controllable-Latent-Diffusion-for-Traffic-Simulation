from setuptools import setup, find_packages
from os import path

this_directory= path.abspath(path.dirname(__file__))
with open(path.join(this_directory,"README.md"),encoding="utf-8") as f:
    lines= f.readlines()

lines= [x for x in lines if ".png" not in x]
long_description="".join(lines)

setup(
    name="tbsim",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    python_requires=">=3.9",
    include_package_data=True,
    # install_requires=[
    #     "pyyaml",
    #     "pytorch-lightning=2.1",
    #     "pyemd",
    #     "h5py",
    #     "imageop-ffmpeg",
    #     "python-louvain",
    #     "nuscenes-devkit",
    #     "protobuf==3.20.*",
    # ],
)
