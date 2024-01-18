#include <ATen/ATen.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
typedef vector<int> T_shape;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(),#x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),#x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void mask_patchcreator(
  const int width,
  float* maskin_ptr,
  // float* maskout_ptr,
  const int M,
  const int maskside,
  const T_shape& Msize,
  float* data_ptr,
  float* patch_ptr,
  // float* patchout_ptr,
  float* center_ptr);

void selection_gpu(
  const int delta,
  float* ratio_ptr,
  const int maskside,
  const int masknum,
  const int M,
  float* maskin_ptr,
  float* patch_ptr,
  float* patchout_ptr,
  float* center_ptr);

torch::Tensor selection(
  torch::Tensor data_im,
  torch::Tensor mask,//5(5*5);7(7*7);11(11*11)
  const int masknum,//number of ==1
  const int delta, //{10,20,40,60,80,100} recess diversity with center
  torch::Tensor ratio //{50%,70%,90%} larger or equal ratiro masked pixels change 
  ){
  CHECK_INPUT(data_im);
  CHECK_INPUT(mask);
  CHECK_INPUT(ratio);
  const int height = data_im.size(0);
  const int width = data_im.size(1);
  T_shape Msize;
  Msize.push_back((height-(mask.size(0)-1)));
  Msize.push_back((width-(mask.size(1)-1)));
  const int M = Msize[0]*Msize[1];
  const int maskside = mask.size(0);
  // auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  // torch::Tensor maskout = torch::empty({M,mask.size(0),mask.size(1),mask.size(2)},mask.options());
  torch::Tensor patch = torch::empty({M,mask.size(0),mask.size(1),3},data_im.options());
  torch::Tensor patchout = torch::empty({3,M,mask.size(0),mask.size(1),3},data_im.options());
  torch::Tensor center = torch::empty({M,3},data_im.options());
  
  // torch::Tensor output = torch::empty({3,data_im.size(0),data_im.size(1),data_im.size(2)},data_im.options());
  // CHECK_INPUT(maskout);
  CHECK_INPUT(patchout);
  // CHECK_INPUT(output);
  auto maskin_ptr = mask.data_ptr<float>();
  // auto maskout_ptr = maskout.data_ptr<float>();
  auto data_ptr = data_im.data_ptr<float>();
  auto patch_ptr = patch.data_ptr<float>();
  auto patchout_ptr = patchout.data_ptr<float>();
  auto center_ptr = center.data_ptr<float>();
  auto ratio_ptr = ratio.data_ptr<float>();
  // auto output_ptr = output.data_ptr<float>();
  mask_patchcreator(
    width,
    maskin_ptr,
    // maskout_ptr,
    M,
    maskside,
    Msize,
    data_ptr,
    patch_ptr,
    // patchout_ptr,
    center_ptr);
  selection_gpu(
    delta,
    ratio_ptr,
    maskside,
    masknum,
    M,
    maskin_ptr,
    patch_ptr,
    patchout_ptr,
    center_ptr);
  return patchout;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
  m.def("selection",&selection,"A module called function to filter pixes");
}




































































