#include <ATen/ATen.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
using namespace std;

//========================pointer method1 dim1======================
// __global__ void selection_kernel(
//   float* data_ptr,
//   float* output_ptr,
//   int N){     
//     int index = blockIdx.x*blockDim.x+threadIdx.x;
//     int stride = blockDim.x*gridDim.x;
//     for (int i =index;i<N;i+=stride){
//       output_ptr[index]=data_ptr[index];
//     }
//   }

// void selection_gpu(
//   float* data_ptr,
//   const int height,
//   const int width,
//   float* output_ptr){
//     int N = height*width*3;
//     int blocksize = 1024;
//     int numbBlocks = (N+blocksize-1)/blocksize;
//     selection_kernel<<<numbBlocks,blocksize>>>(
//       data_ptr,
//       output_ptr,
//       N
//     );
//   }

///====================pointer method2=============================

__global__ void selection_kernel(
  // const int height,
  // const int width,
  float* data_ptr,
  float* output_ptr,
  int N){
      
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int ch = blockIdx.y*blockDim.y+threadIdx.y;
    // int index = col*3+ch;
    if ((col<N)&(ch<3)){
      output_ptr[col*3+ch]=data_ptr[col*3+ch];       
    }

    // int index_row = blockIdx.x*blockDim.x+threadIdx.x;
    // int index_col = blockIdx.y*blockDim.y+threadIdx.y;
    // int index_ch = blockIdx.z*blockDim.z +threadIdx.z;
    // for (int h=index_row;h<height;h+=blockDim.x*gridDim.x){
    //   for (int w=index_col;w<width;w+=blockDim.y*gridDim.y){
    //     // output_ptr[3*(h*height+w)]=data_ptr[3*(h*height+w)];
    //     output_ptr[(h*height+w)*height] =1.1;
    //   }
    // }
    // if((index_row<height) && (index_col<width) && (index_ch<3)){
    //   output_ptr[(index_row*height+index_col)*height-1] =data_ptr[(index_row*height+index_col)*height-1];
    // }
  }

void selection_gpu(
  float* data_ptr,
  const int height,
  const int width,
  float* output_ptr){
    
    int N = height*width;
    const dim3 blocksize(32,32,1);
    int num = (N+blocksize.x-1)/blocksize.x;
    const dim3 numbBlocks(num,1,1);


    // const int threads=1024;
    // const dim3 blocksize(threads,threads,3);
    // int gridrows = (height+blocksize.x -1)/blocksize.x;
    // int gridcols = (width+blocksize.y-1)/blocksize.y;
    // const dim3 numbBlocks(1,1,1);
    selection_kernel<<<numbBlocks,blocksize>>>(
      // height,
      // width,
      data_ptr,
      output_ptr,
      N
    );
  }

///====================pytorch version==================================

// template <typename scalar_t>
// __global__ void selection_gpu_kernel(
//   torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> data_im,
//   torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output_img) {
//   const int width = data_im.size(1);
//   const int height = data_im.size(0);
//   const int wcu =blockIdx.x*blockDim.x+threadIdx.x;
//   if(wcu<width){
//     for(int hcu=0;hcu<height;hcu++){
//       for(int i=0;i<3;i++){
//         output_img[hcu][wcu][i] = data_im[hcu][wcu][i];
//       }
//     }
//   }
// }

// vector<torch::Tensor> selection_gpu(
//   torch::Tensor data_im,
//   torch::Tensor output_img
// ){
//   const int threads = 1024;
//   int gridrows =(data_im.size(0)+threads-1)/threads;
//   AT_DISPATCH_ALL_TYPES(output_img.scalar_type(), "selection_gpu_version", ([&] {
//     selection_gpu_kernel<scalar_t><<<gridrows, threads>>>(
//       data_im.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       output_img.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
//   }));
//   return {output_img};
// }

//====================pytorch v2=================================
// template <typename scalar_t>
// __global__ void selection_gpu_kernel(
//   torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> data_im,
//   torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output_img) {
    
//   const int width = data_im.size(1);
//   const int height = data_im.size(0);
//   const int hcu =blockIdx.x;//blockIdx.x*blockDim.x+threadIdx.x;
//   const int wcu = blockIdx.y;///*blockDim.y+threadIdx.x;
//   const int ch = blockIdx.z;//*blockDim.z+threadIdx.z;
//   // const int stride_w = blockDim.x*gridDim.x;
//   // const int stride_h = blockDim.y*gridDim.y;
//   // for (int i_h=hcu; i_h<data_im.size(0);i_h+=stride_h){
//   //   for(int i_w=wcu; i_w<data_im.size(1); i_w+=stride_w){
//   //     output_img[i_h][i_w][0] = data_im[i_h][i_w][0];
//   //     output_img[i_h][i_w][1] = data_im[i_h][i_w][1];
//   //     output_img[i_h][i_w][2] = data_im[i_h][i_w][2];
//   //   }
//   // }

//   if ((hcu<height) && (wcu<width) && (ch<3)){
//     // for (int i_h=0;i_h<height;i_h++){
//     output_img[hcu][wcu][ch] = data_im[hcu][wcu][ch];
//     // }
//   }
// }

// vector<torch::Tensor> selection_gpu(
//   torch::Tensor data_im,
//   torch::Tensor output_img
// ){
//   const int threads = 1024;
//   const dim3 blocksize(1,1,1);
//   // const dim3 blocksize(threads,threads,1);
//   // int gridrows =(data_im.size(0)+threads-1)/threads;
//   // int gridcols =(data_im.size(1)+threads-1)/threads;
//   const dim3 gridsize(1024,1024,3);

//   AT_DISPATCH_ALL_TYPES(output_img.scalar_type(), "selection_gpu_version", ([&] {
//     selection_gpu_kernel<scalar_t><<<gridsize, blocksize>>>(
//       data_im.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//       output_img.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
//   }));
//   // cudaDeviceSynchronize();
//   return {output_img};
// }








































