from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = 'jmecofftea',
    version = '0.0.1',
    url = 'https://github.com/cms-jet/jme-cofftea',
    author = 'Andreas Albert',
    author_email = 'andreas.albert@cern.ch',
    description = 'JetMET trigger analysis using coffea on NanoAOD',
    packages = find_packages(),    
    install_requires = requirements,
    scripts=[
        'jmecofftea/execute/buexec',
        'jmecofftea/execute/bumon',
        'jmecofftea/scripts/bumerge'
        ],
)
