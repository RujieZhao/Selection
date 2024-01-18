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
  const int width,
  float* maskin_ptr,
  // float* maskout_ptr,
  const int Mw,
  const int M,
  const int maskside,const int maskarea,
  float* data,
  float* patch,
  // float* patchout,
  float* center){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int col = blockIdx.y;//m_h*m_w
    int ch = blockIdx.z;//3
    for (int i = row;i<M;i+=blockDim.x*gridDim.x){
      const int index = (i*maskarea+col)*3+ch;
      // maskout_ptr[index]=maskin_ptr[col*3+ch];
      const int m_w = col % maskside;
      const int m_h = col/maskside;        
      const int c_w = i % Mw;
      const int c_h = i/Mw;
      const int real_c_w = c_w+(maskside/2);
      const int real_c_h = c_h+(maskside/2);
      const int im_w = c_w+m_w;
      const int im_h = c_h+m_h;   
      float* data_ptr = data+((im_h*width+im_w)*3+ch);
      float* patch_ptr = patch+index;
      *patch_ptr=*data_ptr;
      // float result  = (*patch_ptr)*(maskout_ptr[index]);
      // for (int k =0;k<3;k+=1){
      //   patchout[(k*M+i)*maskside*maskside*3+j]=result;
      // }
      float* center_ptr =center + (i*3+ch);
      *center_ptr = data[(real_c_h*width+real_c_w)*3+ch]; 
    }
}

__global__ void selection_kernel(
  float* ratio,
  float* center,
  const int delta,
  const int M,
  const int maskarea,
  const int masknum,
  float* mask,
  float* patch,
  float* patchout){
    __shared__ float tran[32];
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    // int mask_index = blockIdx.y;
    int ch = col%3;
    int m = col/3;
    int patch_index;
    for (int i=col;i<M;i+=blockDim.x*gridDim.x){
      for (int mask_index=0;mask_index<maskarea;mask_index++){
        patch_index = (m*maskarea+mask_index)*3+ch;
        if (mask[mask_index]==1){          
          if (abs(patch[patch_index]-center[col])<delta){
            tran[threadIdx.x]+=1;
          }
        }else{
          patchout[patch_index]=patch[patch_index];
        }  
      }
      __syncthreads();
      if((tran[threadIdx.x]/(float)masknum)>(*ratio)){
        for (int maskindex=0;maskindex<maskarea;maskindex++){
          if(mask[maskindex]==1){
            patch_index = (m*maskarea+maskindex)*3+ch;
            patchout[patch_index] = center[col];
          }
        }
      }else{
        for (int maskindex=0;maskindex<maskarea;maskindex++){
          if(mask[maskindex]==1){
            patch_index = (m*maskarea+maskindex)*3+ch;
            patchout[patch_index] = patch[patch_index];
          }
        }
      }         
    }
  }

void mask_patchcreator(
  // const int height,
  const int width,
  float* maskin_ptr,
  // float* maskout_ptr,
  const int M,
  const int maskside,
  const T_shape& Msize,
  float* data_ptr,
  float* patch_ptr,
  // float* patchout_ptr,
  float* center_ptr){
    const int maskarea = maskside*maskside;
    const int threads=32;
    const int row = (M+threads-1)/threads;
    const dim3 gridsize(row,maskarea,3);
    creator_kernel<<<gridsize,threads>>>(
      width,
      maskin_ptr,
      // maskout_ptr,
      Msize[1],
      M,
      maskside,maskarea,
      data_ptr,
      patch_ptr,
      // patchout_ptr,
      center_ptr);
}

void selection_gpu(
  const int delta,
  float* ratio_ptr,
  const int maskside,
  const int masknum,
  const int M,
  float* maskin_ptr,
  float* patch_ptr,
  float* patchout_ptr,
  float* center_ptr){
    const int threads = 32;
    const int num = (M*3+threads-1)/threads;
    const int maskarea = maskside*maskside;
    const dim3 numbBlocks(num,1,1);
    for (int i=0;i<3;i++){
      selection_kernel<<<numbBlocks,threads>>>(
        ratio_ptr+i,
        center_ptr,
        delta,
        M,
        maskarea,
        masknum,
        maskin_ptr,
        patch_ptr,
        patchout_ptr+i*M*maskarea*3);
    }

  }

































































