#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import sys
from setuptools import setup, find_packages

# Ensure the correct Python version is used
assert sys.version_info >= (3, 6)

# Read the README file for the long description
with open('README.md', 'r', encoding='utf8') as readme_file:
    readme = readme_file.read()

# Read the requirements file
install_reqs = []
with open('requirements_dev.txt', 'r') as f:
    install_reqs = [
        line.strip() for line in f if line.strip() and not line.startswith('#')
    ]

# Separate runtime and development dependencies
runtime_reqs = [
    req for req in install_reqs if not any(
        keyword in req.lower() for keyword in ['pytest', 'redis']
    )
]
dev_reqs = [req for req in install_reqs if 'pytest' in req.lower() or 'redis' in req.lower()]

# Define the setup configuration
setup(
    author="ccafeccafe, Adis Delanovic, Jill Platts, Andre Beckus",
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.9',
    ],
    description="Airlift Challenge Simulator",
    entry_points={
        'console_scripts': [
            'airlift-demo=airlift.cli:demo',
        ],
    },
    install_requires=runtime_reqs,  # Only runtime dependencies
    extras_require={
        'dev': dev_reqs,  # Development and testing dependencies
    },
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='airlift',
    name='airlift-challenge',
    packages=find_packages('.'),
    data_files=[
        ('pngs', [
            'airlift/envs/png/box.png',
            'airlift/envs/png/green_delivered.png',
            'airlift/envs/png/plane.png',
            'airlift/envs/png/red_missed.png',
            'airlift/envs/png/yellow_late.png',
        ]),
    ],
    test_suite='tests',
    url='https://github.com/jomalla123/airlift',  # Replace with your actual repository URL
    version='1.0.0',
    zip_safe=False,
)