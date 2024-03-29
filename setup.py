#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Christopher Bay",
    author_email='christopher.bay@nrel.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Integrated code for determining market participation of wind energy.",
    entry_points={
        'console_scripts': [
            'a2e2g=a2e2g.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='a2e2g',
    name='a2e2g',
    packages=find_packages(include=['a2e2g', 'a2e2g.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/bayc/a2e2g',
    version='0.1.0',
    zip_safe=False,
)
