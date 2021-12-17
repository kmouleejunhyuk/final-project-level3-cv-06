import io
from setuptools import find_packages, setup


# Read in the README for the long description on PyPI
# def long_description():
#     with io.open('README.rst', 'r', encoding='utf-8') as f:
#         readme = f.read()
#     return readme

setup(name='final-project-level3-cv-06',
      version='0.1',
      description='공항 위해 물품 탐지',
    #   long_description=long_description(),
      long_description="나중에",
      url='https://github.com/boostcampaitech2/final-project-level3-cv-06',
      author='AI_Tech_2_CV-06_Team',
      author_email='ppskj178@gmail.com',
      license='unknown',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3.6',
          ],
      zip_safe=False)