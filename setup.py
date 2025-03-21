import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sequent',
    version='0.1.2',    
    description='Python package of SEQUENT, a frequency-based method that employs state machines for the detection of network anomalies.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tudelft-cda-lab/SEQUENT',
    author='Clinton Cao',
    author_email='c.s.cao@tudelft.nl',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['pandas',
                      'numpy',                     
                      ],

    classifiers=[
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)