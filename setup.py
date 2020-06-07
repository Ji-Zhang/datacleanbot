import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datacleanbot",
    version="0.8",
    author="Ji Zhang",
    author_email="",
    description="automated data cleaning tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ji-Zhang/datacleanbot",
    packages=setuptools.find_packages(),
    install_requires=[ 
        'numpy>=1.14.2',
        'pandas>=0.19.0, <=0.23.1',
        'scikit-learn>=0.20.0', 
        'scipy>=1.2.1',
        'seaborn>=0.8',
        'matplotlib>=2.2.2',
        'missingno==0.4.1',
        'openml>=0.9.0',
        'fancyimpute==0.4.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)