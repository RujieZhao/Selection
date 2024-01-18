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
  float* center){
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int col = blockIdx.y;//m_h*m_w
    int ch = blockIdx.z;//3
    for (int i = row;i<M;i+=blockDim.x*gridDim.x){
      const int index = (i*maskarea+col)*3+ch;
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
    int ch = col%3;
    int m = col/3;
    int patch_index;
    for (int i=col;i<(M*3);i+=blockDim.x*gridDim.x){
      tran[threadIdx.x]=0;
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

__global__ void restoration_kernel(
  const int im,
  const int Mh,
  const int Mw,
  const int height,
  const int width,
  const int half,
  const int maskside,
  float* patchout,
  float* output){
    __shared__ int tran[64];
    const int col=threadIdx.x+blockIdx.x*blockDim.x;
    const int ch = col%3;
    const int im_w = (col/3)%width;
    const int im_h = col/3/width;
    for (int i=col;i<im;i+=blockDim.x*gridDim.x){
      tran[threadIdx.x*2]=0;
      tran[threadIdx.x*2+1]=0;
      for (int dh = -half;dh<=half;dh++){
        for(int dw = -half;dw<=half;dw++){
          if((im_h+dh)>=half&&(im_h+dh)<(height-half)&&(im_w+dw)>=half&&(im_w+dw)<(width-half)){
            tran[threadIdx.x*2+1]+=1;
            int center_coor_h=im_h+dh-half;
            int center_coor_w = im_w+dw-half;
            int cur_M = center_coor_h*Mw+center_coor_w;
            int mask_h = half-dh;
            int mask_w = half-dw;
            tran[threadIdx.x*2]+=patchout[((cur_M*maskside+mask_h)*maskside+mask_w)*3+ch];
          }
        }
      }
      __syncthreads();
      output[col] = tran[threadIdx.x*2]/tran[threadIdx.x*2+1];
    } 
  } 

void mask_patchcreator(
  const int width,
  float* maskin_ptr,
  const int M,
  const int maskside,
  const T_shape& Msize,
  float* data_ptr,
  float* patch_ptr,
  float* center_ptr){
    const int maskarea = maskside*maskside;
    const int threads=32;
    const int row = (M+threads-1)/threads;
    const dim3 gridsize(row,maskarea,3);
    creator_kernel<<<gridsize,threads>>>(
      width,
      maskin_ptr,
      Msize[1],
      M,
      maskside,maskarea,
      data_ptr,
      patch_ptr,
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
    // const dim3 numbBlocks(num,1,1);
    for (int i=0;i<3;i++){
      selection_kernel<<<num,threads>>>(
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

void restoration(
  const int Mh,
  const int Mw,
  const int height,
  const int width,
  const int maskside,
  const int M,
  float* patchout,
  float* output){
    const int half = maskside/2;
    const int im=height*width*3;
    const int threads=32;
    const int blocksize = (im+threads-1)/threads;
    for (int i=0;i<3;i++){
      restoration_kernel<<<blocksize,threads>>>(
        im,
        Mh,
        Mw,
        height,
        width,
        half,
        maskside,
        patchout+i*M*maskside*maskside*3,
        output+i*height*width*3);
    }
  }

































































