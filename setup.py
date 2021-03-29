from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

with open('requirements_dev.txt') as f:
    DEV_REQUIREMENTS = f.read().splitlines()

with open('requirements_test.txt') as f:
    TEST_REQUIREMENTS = f.read().splitlines()

EXTRA_REQUIREMENTS = {
    'dev': DEV_REQUIREMENTS,
    'test': TEST_REQUIREMENTS,
}

setup(
    name='synthsonic',
    packages=find_packages(),
    version='0.1.0',
    description='Super realistic data modelling and synthesis',
    author='ING WBAA',
    license='MIT',
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
)
