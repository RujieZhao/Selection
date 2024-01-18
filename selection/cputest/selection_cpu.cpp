#include <ATen/ATen.h>
#include <THC/THC.h>
#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <torch/script.h>

using namespace std;
typedef vector<float> T_shape;

torch::Tensor selection_cpu(
  torch::Tensor data_im
){
  T_shape test;
  // for (int i =0;i<data_im.dim();i++){
  //   test.push_back(data_im.size(i));
  // }
  test.push_back(data_im.size(0));
  test.push_back(data_im.size(1));
  test.push_back(data_im.size(2));
  // auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor shape = torch::from_blob(test.data(),{3});//at::kByte
  return shape.clone();
  // float array[] = { 1, 2, 3, 4, 5};
  // auto options = torch::TensorOptions().dtype(torch::kFloat64);, torch::kFloat32
  // torch::Tensor tharray = torch::from_blob(array, {5});
  // cout<<"tharray:"<<tharray<<endl;
  // return tharray.clone();
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
  m.def("selection_cpu",&selection_cpu,"A module called function to filter pixes");
}

// TORCH_LIBRARY(sel_cpu,m){
//   m.def("selection_cpu",selection_cpu);
// }



// #include <ATen/ATen.h>
// #include <THC/THC.h>
// #include <iostream>
// #include <vector>
// #include <torch/extension.h>
// #include <torch/script.h>

// using namespace std;
// typedef vector<float> T_shape;

// torch::Tensor selection_cpu_cmake(torch::Tensor image) {

//   // float array[] = { 1, 2, 3, 4, 5};
//   // torch::Tensor tharray = torch::from_blob(array, {5});
//   T_shape test;
//   test.push_back(image.size(0));
//   test.push_back(image.size(1));
//   test.push_back(image.size(2));
//   torch::Tensor tharray = torch::from_blob(test.data(), {image.dim()});


//   return tharray.clone();
// }

// TORCH_LIBRARY(sel_cpu, m) {
//   m.def("selection_cpu", selection_cpu_cmake);
// }



























