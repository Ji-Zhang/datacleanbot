import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datacleanbot",
    version="0.4",
    author="Ji Zhang",
    author_email="",
    description="automated data cleaning tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ji-Zhang/datacleanbot",
    packages=setuptools.find_packages(),
    install_requires=[ 
        'numpy>=1.14.2',
        'pandas',
        'scikit-learn>=0.20.0', 
        'scipy>=1.0.0',
        'seaborn>=0.8',
        'matplotlib>=2.2.2',
        'missingno>=0.4.0',
        'fancyimpute',
        'numba>=0.27',
        'pystruct>=0.2.4',
        'cvxopt>=1.1.9',
        'pymc3>=3.4',
        'pyro-ppl>=0.2',
        'rpy2==2.9.4'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)