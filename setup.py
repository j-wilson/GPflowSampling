from setuptools import setup, find_packages

requirements = (
  'numpy>=1.18.0',
  # [!] see: https://github.com/tensorflow/tensorflow/issues/40584
  'tensorflow>=2.2.0',
  'tensorflow-probability>=0.9.0',
  'gpflow>=2.0.3',
)

extra_requirements = {
  'examples': (
    'matplotlib',
    'seaborn',
    'tqdm',
    'tensorflow-datasets',
  ),
}

setup(name='gpflow_sampling',
      version='0.2',
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      packages=find_packages(exclude=["examples*"]),
      python_requires='>=3.6',
      install_requires=requirements,
      extras_require=extra_requirements)
