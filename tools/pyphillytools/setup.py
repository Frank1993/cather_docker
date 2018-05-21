from os import path
from codecs import open
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyphillytools',
    version='1.0.0',
    description='Set of tools for python applications in philly',
    long_description=long_description,
    url='https://philly.visualstudio.com/_git/commonPhilly',
    author='James French',
    author_email='v-jafre@microsoft.com',
    license='Other/Proprietary License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Build Tools',
        'License :: Other/Proprietary License'
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Microsoft',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='pyphillytools philly microsoft',
    packages=find_packages(exclude=['tests'])
)