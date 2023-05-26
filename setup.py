
from setuptools import setup, find_packages

setup(
    name='ditty',
    version='0.3.0',
    license='Apache V2',
    author="Ian T Butler (KinglyCrow)",
    author_email='iantbutler01@gmail.com',
    packages=find_packages('lib'),
    package_dir={'': 'lib'},
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/iantbutler01/ditty',
    keywords='finetuning, llm, nlp, machine learning',
    install_requires=[
          'accelerate',
          'transformers',
          'datasets',
          'bitsandbytes',
          'fire',
          'peft'
      ],

)
