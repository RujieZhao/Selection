#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
typedef vector<int> T_shape;

__device__ int equgross(float* centermap, const int ind, const int width){
  //this model is to get one of duplicate points sum.
  int t=0;
  for(int h=-2;h<3;h++){
    for(int w=-2;w<3;w++){
      t+=centermap[(ind+h*width+w)];
  }}
  return t;
}

__global__ void centermap_kernel(
  float* cloudmap_ptr,
  float* centermap_ptr,
  const int patch,
  const int num_th,
  const int width,
  const int Mw,
  const int M
){
  __shared__ int counter[32];
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  for (int i=ind;i<M;i+=blockDim.x*gridDim.x){
    counter[threadIdx.x]=0;
    const int c_w = i%Mw;
    const int c_h = i/Mw;
    const int real_c_w = c_w+15;
    const int real_c_h = c_h+15;
    int cloud_h,cloud_w,cloud_ind;
    for(int h=-(patch/2); h<=(patch/2);h++){
      for(int w=-(patch/2); w<=(patch/2); w++){
        cloud_h = real_c_h+h;
        cloud_w = real_c_w+w;
        cloud_ind = cloud_h*width+cloud_w;
        if(cloudmap_ptr[cloud_ind]!=0){
          counter[threadIdx.x]+=cloudmap_ptr[cloud_ind];
    }}}
    __syncthreads();
    if (counter[threadIdx.x]>=num_th){
      int centermap_ind = real_c_h*width+real_c_w;
      centermap_ptr[centermap_ind]=counter[threadIdx.x];
}}}

__global__ void equlist_kernel(
  float* finalcentermap,
  float* centermap,
  unsigned int* equlist,
  unsigned int* grossmap,
  const int width,
  const int M,
  const int Mw
  // const int patch
){
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  for (int i =ind;i<M;i+=blockDim.x*gridDim.x){
    const int cw = i%Mw;
    const int ch = i/Mw;
    const int real_w = cw+15;
    const int real_h = ch+15;
    const int real_ind = real_h*width+real_w;
    const int cv = centermap[real_ind];
    int centermap_ind=0;
    //a)create equlist which represents those equivalent points with respect to every centers.
    //b)restore those equal position to a grossmap that is for avoiding duplicate sum calcuation.
    //c)After euqlist acomplished, for all unzero vlaues in centermap, set a mask on final centermap to mark those unique postions by the oppsite equlist.

    // printf("this is for test equlist initializtion: No. %d, the euqlist value is %d, the grossmap value is %d, the size of equ: %d\n",ind,equlist[ind],grossmap[ind],sizeof(equlist[ind]));

    if(cv!=0){
      for (int h=-2;h<=2;h++){
        for (int w=-2;w<=2;w++){
          centermap_ind = real_ind+h*width+w;
          //if i account for h=0&w=0 then every centers will be counted for 1 even when no one equals to it, that is not expected. And for those centers have duplicate subordinates, we add their loc to equlist after the loop. Grossmap contains every euqivalent locs including centers.
          if((h!=0)|(w!=0)&&(cv==centermap[centermap_ind])){
            equlist[i]|=1<<((h+2)*5+(w+2));
            int grossmap_ind = i+h*Mw+w;
            if(grossmap[grossmap_ind]==0){grossmap[grossmap_ind]=equgross(centermap,centermap_ind,width);}
      }}}
      if(equlist[i]!=0){equlist[i]|=1<<12;}
      else if(equlist[i]==0){finalcentermap[real_ind]=1;}
      // printf("after,No: %d; equlist value: %d; grossmap value: %d \n",i,equlist[i],grossmap[i]);
}}}

__global__ void grosscomp(
  float* cloudmap,
  float* finalcentermap,
  unsigned int* equlist,
  unsigned int* grossmap,
  unsigned int* tranmap,
  // const int patch,
  const int width,
  const int M,
  const int Mw
){
  //2. this is following equlist. Considering grossmap and equlist I created in euqlist part, doing comparison process.
  //a, amongst those equal position, pick the one with largest gross value fill to the tranmap;
  //b, If those chosen postion in a) is not unique, then use cloudmap to pick one with largest value;
  //c, If in b), the position still not unique, then use the location with lower indices;

  int ind  = blockIdx.x*blockDim.x+threadIdx.x;
  int w,h,pre_state=0,curr_cloud=0,curr_gro=0,count=0,dw=0,dh=0,nw=0,nh=0,ind_map,index;
  for(int i=ind;i<M;i+=blockDim.x*gridDim.x){
    w = i%Mw;h = i/Mw;
    //this is for verdict wheather current center has equivalent subsequence
    if (grossmap[i]!=0){//grossmap contains every qualified centers.
      for (index=0;index<25;index++){
        if((1<<index)&equlist[i]){
          if (index!=12){
            dw = index%5-2; dh = index/5-2;
            nw = w+dw; nh = h+dh;
            ind_map = nh*Mw+nw;
          }else{ind_map=i;}
          if(grossmap[ind_map]>curr_gro){
            curr_gro=grossmap[ind_map];count=1;tranmap[i] = (1<<index);
          }else if(grossmap[ind_map]==curr_gro){
            count+=1;tranmap[i]|=(1<<index);
          }}}
      // printf("in(a) show tranmap: %d; count: %d \n",tranmap[ind], count);
      //here is section b, use cloudmap to filter out centers.
      if(count!=1){
        count=0;
        for(index=0;index<25;index++){
          pre_state|=1<<index;
          if((1<<index)&tranmap[i]){
            if(index!=12){
              dw = index%5-2; dh = index/5-2;
              nw = w+dw+15; nh = h+dh+15;
              ind_map = nh*width+nw;
            }else{ind_map=(h+15)*width+w+15;}
            if(count==0){curr_cloud=cloudmap[ind_map];};
            if(cloudmap[ind_map]<curr_cloud){
              tranmap[i]&=~(1<<index);
            }else if(cloudmap[ind_map]==curr_cloud){count+=1;}
            else if(cloudmap[ind_map]>curr_cloud){
              curr_cloud=cloudmap[ind_map];count=1;
              tranmap[i]&=~pre_state;tranmap[i]|=(1<<index);}
      }}}
      //section c, use less index to pick excusive center out of equalivant
      if(count>=2){
        for(index=0;index<25;index++){
          if((1<<index)&tranmap[i]){tranmap[i]=(1<<index);count=1;break;
        }}}
      if((1<<12)&tranmap[i]){finalcentermap[((h+15)*width+w+15)]=1;}
    }}
    // printf("tranmap: %d; count: %d\n",tranmap[ind],count);
  __syncthreads();
}

__global__ void restoration(
  unsigned int* tranmap,
  unsigned int* equlist,
  float* centermap,
  float* finalcenter,
  const int width,
  const int M,
  const int Mw
){
  //3.transfer centermap's value to finalcenter to eliminate equivalent problem and narrow the range.
  //a)if the current value shows in equlist, then start judgement.
  //b)if the same value whose postion doesnt show up in tranmap, then transfer 0 to finalcenter instead orignal value.
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  int w,h,ind_map;
  for(int i = ind;i<M;i+=blockDim.x*gridDim.x){
    w = i%Mw; h = i/Mw; ind_map=(h+15)*width+w+15;
    if(centermap[ind_map]!=0){finalcenter[ind_map]=finalcenter[ind_map]*centermap[ind_map];}
}}  

__global__ void pyramid_kernel(
  float* cloudmap,
  float* centermap,
  float* finalcenter,
  const int patch,
  const int pylayer,
  const int num_ignore,
  const float amp,
  const int width,//orignal pic width/cloudmap width
  const int M,
  const int Mw
){
  /*this kernel is mainly for building pyramid structure. It aim at change and filter each current center. The first layer of pyramid should be all smaller than the center's value. In second layer, it is allowed to have "num_ignore" value to larger than center value and within amp percent. If above conditions matches current center, then the center will be perserved as a valid one.*/
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  int h,w,real_h,real_w,mapind,cv,layer,nind,dh,dw,count=0;
  for(int i=ind;i<M;i+=blockDim.x*gridDim.x){
    h = i/Mw; w = i%Mw; real_h = h+15; real_w = w+15; 
    mapind = real_h*width+real_w;
    cv = finalcenter[mapind];
    // if(mapind==55453){printf("55453: %d \n",cv)}
    if(cv!=0){
      // printf("finalcenter value: %d, h: %d, w: %d\n",cv,real_h,real_w);
      for(layer=0;layer<=pylayer;layer++){
        if(layer==0){
          for(dh=-1; dh<=1; dh++){
            for(dw=-1; dw<=1; dw++){
              if(((dh!=0)|(dw!=0))){
                nind = mapind+dh*width+dw;
                if(centermap[nind]>cv){finalcenter[mapind]=0;goto stop;}
        }}}
      }else if(layer==1){
          int d[]={-2,2};
          for(int z:d){
            for(dh=-2;dh<=2;dh++){
              nind = mapind+dh*width+z;
              if(centermap[nind]>cv){ count+=1;
                if((count>num_ignore)||(((centermap[nind]-cv)/cv)<=amp)){finalcenter[mapind]=0;goto stop;}
              }}
            for(dw=-1;dw<=1;dw++){
              nind = mapind+z*width+dw;
              if(centermap[nind]>cv){count+=1;
                if((count>num_ignore)||(((centermap[nind]-cv)/cv)<=amp)){finalcenter[mapind]=0;goto stop;}
    }}}}}stop: 
    // printf("cv:%d\n",cv);((((finalcenter[nind]-cv)/cv)<=amp)&&((finalcenter[nind]-cv)/cv)>=0))
  }}}

__global__ void finalcloud_kernel(
  float* cloudmap,
  float* finalcloud,
  float* finalcenter,
  const int width,
  const int M,
  const int patch,
  const int Mw
){
  int ind = blockIdx.x*blockDim.x+threadIdx.x;
  int h,w,real_c_h,real_c_w,mapind,dh,dw,cv,newind;
  for (int i=ind; i<M; i+=blockDim.x*gridDim.x){
    w=i%Mw; h=i/Mw; real_c_h=h+15; real_c_w=w+15; 
    mapind=real_c_h*width+real_c_w;cv=finalcenter[mapind];
    if(cv>0){
      for(dh=-(patch/2);dh<=patch/2;dh++){
        for(dw=-(patch/2);dw<=patch/2;dw++){
          newind=mapind+dh*width+dw;
          if(cloudmap[newind]!=0){finalcloud[newind]=cloudmap[newind];}
}}}__syncthreads();
}}

void center_map_creator(
  float* cloudmap_ptr,
  float* centermap_ptr,
  const int M,
  const int patch,
  const int width,
  const T_shape&Msize,
  const int num_th
){
  const int threads=32;
  const int block=(M+threads-1)/threads;
  centermap_kernel<<<block,threads>>>(
    cloudmap_ptr,
    centermap_ptr,
    patch,
    num_th,
    width,
    Msize[1],
    M
  );
}

void equivalent(
  // unsigned int* grossmapcpu,
  float* centermap_ptr,
  float* cloudmap_ptr,
  float* finalcenter_ptr,
  // float* testmap_ptr,
  const int M,
  const int width,
  const T_shape& Msize
){
  const int threads = 32;
  const int block = (M+threads-1)/threads;
  const size_t bytes = sizeof(unsigned int)*M;//size_t:unsigned long
  unsigned int *equlist,*grossmap,*tranmap;
  cudaMalloc((void**)&equlist,bytes);
  cudaMemset(equlist,0,bytes);
  cudaMalloc((void**)&grossmap,bytes);
  cudaMemset(grossmap,0,bytes);
  cudaMalloc((void**)&tranmap,bytes);
  cudaMemset(tranmap,0,bytes);
  //1.find out those equivalent coordinates,fill one to unique unzero centers on finalcentermap
  equlist_kernel<<<block,threads>>>(
    finalcenter_ptr,
    centermap_ptr,
    equlist,
    grossmap,
    width,
    M,
    Msize[1]
  );

  //2.few steps of compairison with grossmap and cloudmap to obtain an exclusive position on tranmap for every equivalent patch and pass "1" to finalcentermap.
  grosscomp<<<block,threads>>>(
    cloudmap_ptr,
    finalcenter_ptr,
    equlist,
    grossmap,
    tranmap,
    width,
    M,
    Msize[1]
  );

  // cudaMemcpy(grossmap,grossmapcpu,bytes,cudaMemcpyDeviceToHost);
  //3. Use the tranmap fill out the finalcentermap;
  restoration<<<block,threads>>>(
    tranmap,
    equlist,
    centermap_ptr,
    finalcenter_ptr,
    width,
    M,
    Msize[1]
  );
  // cudaMemcpy(testmap_ptr,finalcenter_ptr,bytes,cudaMemcpyDeviceToHost);
  cudaFree(equlist);
  cudaFree(grossmap);
  cudaFree(tranmap);
}

void pyramid(
  float* cloudmap_ptr,
  float* centermap_ptr,
  float* finalcenter_ptr,
  const int width,
  const int M,
  const T_shape& Msize,
  const int patch,
  const int pylayer,
  const int num_ignore,
  const float amp
){
  const int threads = 32;
  const int block = (M+threads-1)/threads;
  /*In the pyramid kernel, we only foucs on each centers and its adjunct locs are references, we will not do any operations on them.*/
  pyramid_kernel<<<block,threads>>>(
    cloudmap_ptr,
    centermap_ptr,
    finalcenter_ptr,
    patch,
    pylayer,
    num_ignore,
    amp,
    width,
    M,
    Msize[1]
  );
}

void finalcloud_creator(
  float* cloudmap_ptr,
  float* finalcloud_ptr,
  float* finalcenter_ptr,
  const int width,
  const int M,
  const T_shape& Msize,
  const int patch
){
  const int threads = 32;
  const int block = (M+threads-1)/threads;
  finalcloud_kernel<<<block,threads>>>(
    cloudmap_ptr,
    finalcloud_ptr,
    finalcenter_ptr,
    width,
    M,
    patch,
    Msize[1]
  );
}











