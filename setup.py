from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='SMA',
    version='0.1.0',
    description='SMA Project',
    author='SMA Team of DSSGx 2021',
    author_email='dssg@wbs.ac.uk',
    long_description=readme,
    url='https://gitlab.uksouth.saas.aridhia.io/u410127/sma',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

