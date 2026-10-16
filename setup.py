import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmix",  # Replace with your prefered name
    version="0.0.1",
    author="mblons",
    author_email="matthieu.blons@hotmail.fr",
    description="A package for whole-slide images (WSI) processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={"cosmix.bioformats": ["configs/*.yaml", "configs/*.xml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
