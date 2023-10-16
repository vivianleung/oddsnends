#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Vivian Leung",
    author_email='leung.vivian.w@gmail.com',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="Miscellaneous data science methods",
    entry_points={
        # 'console_scripts': [
        #     'oddsnends=oddsnends.cli:main',
        # ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='oddsnends',
    name='oddsnends',
    packages=find_packages(include=['oddsnends', 'oddsnends.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/vivianleung/oddsnends',
    version='0.1.0',
    zip_safe=False,
)
