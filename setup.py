import setuptools

# Read the version from the dorna_vision package
version = "2.4.0"

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

# Read the requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Setup function to include the requirements
setuptools.setup(
    name="dorna_vision",
    version=version,
    author="Dorna Robotics",
    author_email="info@dorna.ai",
    description="Dorna vision package",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://dorna.ai/",
    project_urls={
        "gitHub": "https://github.com/dorna-robotics/dorna_vision",
    },
    packages=setuptools.find_packages(),
    install_requires=requirements,  # This ensures requirements.txt is used
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
