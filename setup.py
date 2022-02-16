import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as r:
    requires = [ln.replace("\n", "").strip() for ln in r.readlines()]

setuptools.setup(
    name="config4ml",
    version="0.0.4",
    author="Bill Tsironis",
    author_email="tsironisbi@gmail.com",
    description="Basic configuration utilities and API for ML projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsironisbi/config4ml/",
    project_urls={
        "Bug Tracker": "https://github.com/tsironisbi/config4ml/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=requires,
)
