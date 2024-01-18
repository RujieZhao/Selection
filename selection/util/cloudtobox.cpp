#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>

using namespace std;
typedef vector<int> T_shape;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(),#x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),#x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void cloudcomp(
  float* finalcloud,
  float* cloud_indice,
  float* orignal_coor,
  const int M,
  const T_shape& imgsize,
  const int len_indice
);

void selecbox(
  float* finalcloud,
  float* finalcenterlist,
  float* outputbox,
  const int hpatch,
  const T_shape& imgsize,
  const int boxnum
);

torch::Tensor boxcreator(
  const int patch,
  torch::Tensor finalcloud,
  torch::Tensor finalcenterlist,
  torch::Tensor cloud_indice,
  torch::Tensor orignal_coor
){
  CHECK_INPUT(finalcloud);
  CHECK_INPUT(finalcenterlist);
  CHECK_INPUT(cloud_indice);
  CHECK_INPUT(orignal_coor);
  
  const int hpatch = patch/2;
  const int len_indice = cloud_indice.size(0);
  // printf("len_indice:%d \n",len_indice);
  const int imgh = finalcloud.size(0);
  const int imgw = finalcloud.size(1);
  // printf("size: %d,%d\n",imgh,imgw);
  T_shape imgsize;
  imgsize.push_back(imgh);
  imgsize.push_back(imgw);
  const int M = imgsize[0]*imgsize[1];
  const int boxnum = finalcenterlist.size(0);
  // printf("box num: %d \n",boxnum);
  torch::Tensor outputbox = torch::zeros({boxnum,4},finalcenterlist.options());

  auto outputbox_ptr = outputbox.data_ptr<float>();
  auto finalcloud_ptr = finalcloud.data_ptr<float>();
  auto finalcenterlist_ptr = finalcenterlist.data_ptr<float>();
  auto cloud_indice_ptr = cloud_indice.data_ptr<float>();
  auto orignal_coor_ptr = orignal_coor.data_ptr<float>();
  
  /*cloud comp is to fill the finalcoud[img_h,img_w,5] with the largest orignal corrs value(min_h,min_w,max_h,max_w)*/
  cloudcomp(
    finalcloud_ptr,
    cloud_indice_ptr,
    orignal_coor_ptr,
    M,
    imgsize,
    len_indice
  );

  selecbox(
    finalcloud_ptr,
    finalcenterlist_ptr,
    outputbox_ptr,
    hpatch,
    imgsize,
    boxnum
  );

  return outputbox;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("boxcreator",&boxcreator,"this mode integrates cpu c++ and gpu cuda to transfer finalcloud to the final box conresponding to each finalcenters.");
}














