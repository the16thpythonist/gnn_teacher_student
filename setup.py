#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    content = requirements_file.read()
    requirements = content.splitlines()


test_requirements = [ ]

setup(
    author="Jonas Teufel",
    author_email='jonseb1998@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Implementation for the teacher student analysis to evaluate the quality of graph attention explanations",
    entry_points={
        'console_scripts': [
            'gnn_teacher_student=gnn_teacher_student.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gnn_teacher_student',
    name='gnn_teacher_student',
    packages=find_packages(include=['gnn_teacher_student', 'gnn_teacher_student.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/the16thpythonist/gnn_teacher_student',
    version='0.1.0',
    zip_safe=False,
)
