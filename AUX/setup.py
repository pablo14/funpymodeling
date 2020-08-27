import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'funPyModeling'
AUTHOR = 'You'
AUTHOR_EMAIL = 'pcasas.biz@gmail.com'
URL = 'https://github.com/pablo14/funPyModeling'

LICENSE = 'MIT'
DESCRIPTION = 'A  package designed for data scientists and teachers, to speed up their ML projects, focused on exploratory data analysis, data preparation, and model performance.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'matplotlib',
      'sklearn',
      'seaborn'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )


