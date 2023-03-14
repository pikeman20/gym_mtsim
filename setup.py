from setuptools import setup, find_packages

setup(
    name='gym_mtsim',
    version='1.2.0',
    packages=find_packages(),

    author='AminHP',
    author_email='mdan.hagh@gmail.com',

    install_requires=[
        'gym',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'plotly',
        'nbformat',
        'pathos',
    ],

    package_data={
        'gym_mtsim': ['data/*.pkl']
    }
)
