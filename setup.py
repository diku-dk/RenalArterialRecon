from setuptools import setup, find_packages


with open("requirements.txt") as req_file:
    requirements = list(filter(None, req_file.read().split("\n")))


setup(
    name='VesselGen',
    version=1,
    packages=find_packages(),
    package_dir={'VesselGen':
                 'VesselGen'},
    include_package_data=True,
    
    install_requires=requirements,
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'Operating System :: POSIX',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: MIT License']
)
