# from distutils.core import setup,find_packages
from setuptools import find_packages, setup

setup(name='torchdistpackage',
      version='0.1',
      description='TorchDistPackage',
      author='KimmiShi',
      author_email='',
      url='https://github.com/KimmiShi/TorchDistPackage',
      packages=find_packages(include=['torchdistpackage', 'torchdistpackage.*'])
     )