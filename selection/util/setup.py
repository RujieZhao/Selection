
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,load

setup(
  name="selection_pyramid",
  ext_modules=[
    CUDAExtension("selecpycuda",[
    "pyramid.cpp",
    "pyramid_cuda.cu",
    ])
  ],
  cmdclass={
  "build_ext":BuildExtension.with_options(no_python_abi_suffix=True)
  }
)

setup(
  name="selecbox",
  ext_modules=[
    CUDAExtension("selecbox",[
      "cloudtobox.cpp",
      "cloudtobox_cuda.cu",
    ])
  ],
  cmdclass={
    "build_ext":BuildExtension.with_options(no_python_abi_suffix=True)
  }
)


# ext_modules =[
# 	Extension(
# 		"contour",
# 		sources = ["contour.pyx"],
# 		include_dirs= [np.get_include(),],
# 		extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "std=c99"],
# 	)
# ]
#
# setup(
# 	name='selection IS target',
# 	# packages=['util'],
# 	# package_dir = {'util': 'util'},
# 	install_requires=[
# 		'setuptools>=18.0',
# 		'cython>=0.27.3',
# 	],
# 	version='2.0',
# 	ext_modules= ext_modules,
# )







