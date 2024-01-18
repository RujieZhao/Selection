
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
using namespace std;
typedef vector<int> T_shape;

__global__ void creator_kernel(
  // const int height, 
  const int width,
  float* maskin_ptr,
  float* maskout_ptr,
  const int Mh,const int Mw,
  const int M,
  const int maskside,
  float* data,
  float* patch){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int col = blockIdx.y;//m_h*m_w
    int ch = blockIdx.z;//3
    for (int i = row;i<M;i+=blockDim.x*gridDim.x){
      const int index = (i*maskside*maskside+col)*3+ch;
      maskout_ptr[index]=maskin_ptr[col*3+ch]; 
      const int c_w = i % Mw;
      const int c_h = i/Mw;
      const int m_w = col % maskside;
      const int m_h = col/maskside;
      const int im_w = c_w+m_w;
      const int im_h = c_h+m_h;   
      float* data_ptr = data+((im_h*width+im_w)*3+ch);
      float* patch_ptr = patch+index;
      *patch_ptr=*data_ptr;
    }
}

__global__ void selection_kernel(
  // const int height,
  // const int width,
  float* data_ptr,
  float* output_ptr,
  int N){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int ch = threadIdx.y;
    if ((col<N)){
      output_ptr[col*3+ch]=data_ptr[col*3+ch];
    }
  }

void mask_patchcreator(
  // const int height,
  const int width,
  float* maskin_ptr,
  float* maskout_ptr,
  const int M,
  const int maskside,
  const T_shape& Msize,
  float* data_ptr,
  float* patch_ptr){
    const dim3 blocksize(64,1,1); //(64,10,1)
    const int row = (M+blocksize.x-1)/blocksize.x;
    const int maskarea = maskside*maskside;
    const dim3 gridsize(row,maskarea,3);
    creator_kernel<<<gridsize,blocksize>>>(
      // height,
      width,
      maskin_ptr,
      maskout_ptr,
      Msize[0], Msize[1],
      M,
      maskside,
      data_ptr,
      patch_ptr
    );
}

void selection_gpu(
  float* data_ptr,
  const int height,
  const int width,
  float* output_ptr){
    int N = height*width;
    const dim3 blocksize(32,3,1);
    int num = (N+blocksize.x-1)/blocksize.x;
    const dim3 numbBlocks(num,1,1);
    selection_kernel<<<numbBlocks,blocksize>>>(
      data_ptr,
      output_ptr,
      N
    );
  }


































