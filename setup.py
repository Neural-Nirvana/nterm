from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nterm",
    version="0.2.0",
    author="Neural Nirvana",
    author_email="ekansh@duck.com",
    description="A Terminal based AI reasoning agent with system administration and IoT capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Neural-Nirvana/nterm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "nterm=nterm.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)