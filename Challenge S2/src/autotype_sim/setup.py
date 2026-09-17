import os
from glob import glob

from setuptools import find_packages, setup

package_name = "autotype_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "web"), glob(f"{package_name}/web/*")),
        (os.path.join("share", package_name, "assets"), glob(f"{package_name}/assets/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Rishav Nair",
    maintainer_email="rishav.a.nair@gmail.com",
    description="URC autonomous-typing challenge simulator.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"sim_node = {package_name}.nodes.sim_node:main",
        ],
    },
)
