import setuptools
import dorna_vision

# Read the version from the dorna_vision package
version = dorna_vision.__version__

# Read the contents of README.md
with open("README.md", "r") as fh:
    readme = fh.read()

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Setup function to include the requirements
setuptools.setup(
    name="dorna_vision",
    version=version,
    author="Dorna Robotics",
    author_email="info@dorna.ai",
    description="Dorna vision package",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://dorna.ai/",
    project_urls={
        'gitHub': 'https://github.com/dorna-robotics/dorna_vision',
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
