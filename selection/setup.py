


from torch.utils.cpp_extension import load
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])
# setup(name='lltm_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})


# Extension(
#    name='lltm_cpp',
#    sources=['lltm.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++')



# setup(
#     name='lltm',
#     ext_modules=[
#         CUDAExtension('lltm_cuda', [
#             'lltm_cuda.cpp',
#             'lltm_cuda_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         # 'build_ext': BuildExtension
#         "build_ext":BuildExtension.with_options(no_python_abi_suffix=True)
#     })


# lltm = load(name='lltm', sources=['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])

"""================================================================================="""

# setup(
#     name="selection",
#     ext_modules=[cpp_extension.CppExtension("selcuda",["selection.cpp"])],
#     cmdclass={"build_ext":BuildExtension.with_options(no_python_abi_suffix=True)}
# )

setup(
    name="selection",
    ext_modules=[
        CUDAExtension("selcuda",[
            "selection.cpp",
            "selection_cuda.cu",
        ])
    ],
    cmdclass={
        "build_ext":BuildExtension.with_options(no_python_abi_suffix=True)
    })

# setup(name='selection',
#       ext_modules=[CUDAExtension('selcuda',['selection.cpp','selection_cuda.cu'])],
#       cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)})

# setup(name='selection_cpu',
#       ext_modules=[cpp_extension.CppExtension('selcpu',["selection_cpu.cpp"])],
#       cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)})







