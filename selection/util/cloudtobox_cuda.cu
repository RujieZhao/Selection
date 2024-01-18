#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
using namespace std;
typedef vector<int> T_shape;

__global__ void cloudcompkernel(
  float* cloud_indice,
  float* finalcloud,
  float* orignal_coor,
  const int M,
  const int img_height, const int img_width,
  const int len_indice
){
  const int ind = blockDim.x*blockIdx.x+threadIdx.x;
  for(int i=ind;i<M;i+=blockDim.x*gridDim.x){
    int cv = finalcloud[i*5];
    if(cv !=0){
      // printf("valid cloud points:%d, %d\n",i,cv);
      int count=0,w=i%img_width,h=i/img_width,max_h=-1,max_w=-1,min_h=img_height,min_w=img_width;
      for(int indice=0;indice<len_indice;indice++){
        if((cloud_indice[indice*2]==h)&&(cloud_indice[indice*2+1]==w)){
          count++;
          if(orignal_coor[indice*2]>max_h){max_h=orignal_coor[indice*2];}
          if(orignal_coor[indice*2]<min_h){min_h=orignal_coor[indice*2];}
          if(orignal_coor[indice*2+1]>max_w){max_w=orignal_coor[indice*2+1];}
          if(orignal_coor[indice*2+1]<min_w){min_w=orignal_coor[indice*2+1];}
          if(count==cv){
            // printf("indice: %d\n",indice);
            goto stop;}
        }
      }stop:
      // printf("min and max hw: %d,%d,%d,%d\n",min_h,min_w,max_h,max_w);
      finalcloud[i*5+1]=min_h;finalcloud[i*5+2]=min_w;
      finalcloud[i*5+3]=max_h;finalcloud[i*5+4]=max_w;
}}}

__global__ void selecboxkernal(
  float* finalcloud,
  float* finalcenterlist,
  float* outputbox,
  const int hpatch,
  const int img_height,const int img_width,
  const int boxnum 
){
  // printf("size:%d,%d\n",img_height,img_width);
  int cloud_ind,min_h=img_height,min_w=img_width,max_h=-1,max_w=-1,count=0;
  const int ind = blockIdx.x*blockDim.x+threadIdx.x;
  for (int i=ind;i<boxnum;i+=gridDim.x*blockDim.x){
    int cv = finalcenterlist[i*2];
    // printf("valid finalcenter points:%d, %d,%d\n",i,cv,boxnum);
    int center_ind = finalcenterlist[i*2+1];
    // printf("centerid:%d,%d\n",center_ind,hpatch);
    for(int cloudh=-hpatch;cloudh<=hpatch;cloudh++){
      for(int cloudw=-hpatch;cloudw<=hpatch;cloudw++){
        cloud_ind=center_ind+cloudh*img_width+cloudw;
        if(finalcloud[cloud_ind*5]!=0){
          count++;
          if(finalcloud[cloud_ind*5+1]<min_h) min_h=finalcloud[cloud_ind*5+1];
          if(finalcloud[cloud_ind*5+2]<min_w) min_w=finalcloud[cloud_ind*5+2];
          if(finalcloud[cloud_ind*5+3]>max_h) max_h=finalcloud[cloud_ind*5+3];
          if(finalcloud[cloud_ind*5+4]>max_w) max_w=finalcloud[cloud_ind*5+4];
        }
        if(count==cv) goto stop;
    }}stop:
    //XYXY_ABS pic's top-left and bottom-right;
    outputbox[i*4]=min_w;outputbox[i*4+1]=min_h;outputbox[i*4+2]=max_w;outputbox[i*4+3]=max_h;
  }
}


void cloudcomp(
  float* finalcloud,
  float* cloud_indice,
  float* orignal_coor,
  const int M,
  const T_shape& imgsize,
  const int len_indice
){
  const int threads =32;
  const int block = (M+threads-1)/threads;
  cloudcompkernel<<<block,threads>>>(
    cloud_indice,
    finalcloud,
    orignal_coor,
    M,
    imgsize[0],imgsize[1],
    len_indice
  );
}

void selecbox(
  float* finalcloud,
  float* finalcenterlist,
  float* outputbox,
  const int hpatch,
  const T_shape& imgsize,
  const int boxnum
){
  const int threads=32;
  const int block = (boxnum+threads-1)/threads;
  selecboxkernal<<<block,threads>>>(
    finalcloud,
    finalcenterlist,
    outputbox,
    hpatch,
    imgsize[0],imgsize[1],
    boxnum
  );
}




















