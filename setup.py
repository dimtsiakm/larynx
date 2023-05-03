from setuptools import find_packages, setup

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    name='larynx',
    version='0.1.0',
    description='A medical imaging segmentation using DECT dataset',
    author='Dimitris Tsiakmakis',
    license='MIT',
)
