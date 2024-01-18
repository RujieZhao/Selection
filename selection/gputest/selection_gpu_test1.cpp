#include <ATen/ATen.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <tuple>


using namespace std;
typedef vector<int> T_shape;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(),#x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),#x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void selection_gpu(
  float* data_ptr,
  const int height,
  const int width,
  float* output_ptr
  );

torch::Tensor selection(
  torch::Tensor data_im
  ){
  CHECK_INPUT(data_im);
  const int height = data_im.size(0);
  const int width = data_im.size(1);
  // auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor output = torch::empty({data_im.size(0),data_im.size(1),data_im.size(2)},data_im.options());
  CHECK_INPUT(output);
  auto data_ptr = data_im.data_ptr<float>();
  auto output_ptr = output.data_ptr<float>();
  
  selection_gpu(
    data_ptr,
    height,
    width,
    output_ptr);
  return output;
}

//==========================pytorch version============================
// vector<torch::Tensor> selection_gpu(
//   torch::Tensor data_im,
//   torch::Tensor output
// );

// vector<torch::Tensor> selection(
//   torch::Tensor data_im
// ){
//   CHECK_INPUT(data_im);
//   auto output = torch::empty({data_im.size(0),data_im.size(1),data_im.size(2)},data_im.options());
//   return selection_gpu(data_im,output);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
  m.def("selection",&selection,"A module called function to filter pixes");
}
















































