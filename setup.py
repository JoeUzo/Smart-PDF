from setuptools import setup, find_packages

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="Smart-PDF",
    version="0.1.0",
    author="Joe",
    description="A smart PDF processing tool",
    packages=find_packages(),
    install_requires=requirements,
)