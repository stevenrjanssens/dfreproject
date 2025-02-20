from setuptools import setup, find_packages

setup(
    name="reprojection",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    exclude=["tests.*", "tests", "docs.*", "docs"]
)