import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoth-qiuqiangkong", # Replace with your own username
    version="0.0.2",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="Automatic threshold optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuqiangkong/autoth",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)