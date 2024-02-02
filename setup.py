import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name="splice",
    version="1.0",
    description="",
    author="Alex Oesterling, Usha Bhalla",
    author_email="aoesterling@g.harvard.edu, usha_bhalla@g.harvard.edu",
    py_modules=["splice"],
    packages=find_packages(exclude=["experiments*", "data*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)