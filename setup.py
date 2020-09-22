from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [x.strip() for x in f.readlines() if len(x.strip()) > 0 and not x.strip().startswith('-')]

setup(
    name='synthsonic',
    packages=find_packages(),
    version='0.1.0',
    description='Super realistic data modelling and synthesis',
    author='ING WBAA',
    license='MIT',
    install_requires=requirements
)
