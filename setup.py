import setuptools
from io import open
from os import path

# Get the long description from the README file
with open('README.md','r') as f:
    long_description = f.read()


setuptools.setup(
	name='mccv',
	version='0.1a0',
	description='Monte Carlo Cross Validation',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='',
	author='Nicholas Giangreco',
	author_email='nick.giangreco@gmail.com',
	classifiers=[
	'Development Status :: 3 - Alpha',
	'Programming Language :: Python :: 3'
	],
	keywords='machine learning cross validation prediction',
	license='MIT',
	packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests']),
	install_requires=[
	'numpy',
	'pandas',
	'scikit-learn',
	'joblib'
	],
	test_suite='nose.collector',
	tests_require=['nose'],
	zip_safe=False)
