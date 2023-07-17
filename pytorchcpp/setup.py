from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="torch_eval",
      ext_modules=[cpp_extension.CppExtension("torch_eval", ["main.cpp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
