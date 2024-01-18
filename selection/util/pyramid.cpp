#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
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

void center_map_creator(
  float* cloudmap_ptr,
  float* centermap_ptr,
  const int M,
  const int patch,
  const int width,
  const T_shape& Msize,
  const int num_th
);

void equivalent(
  // unsigned int* grossmapcpu,
  float* centermap_ptr,
  float* cloudmap_ptr,
  float* finalcenter_ptr,
  // float* testmap_ptr,
  const int M,
  const int width,
  const T_shape& Msize
);

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
);

void finalcloud_creator(
  float* cloudmap_ptr,
  float* finalcloud_ptr,
  float* finalcenter_ptr,
  const int width,
  const int M,
  const T_shape& Msize,
  const int patch
);

vector<torch::Tensor> selecpy(
  torch::Tensor cloudmap, //[img_h,img_w]
  // torch::Tensor valid_indices,//coordinates in cloudmap whose value is not zero [N,2]
  const int patch, //7
  const int pylayer, //2
  const int num_ignore, //1
  const float amp, //0.2
  const int num_th //6
){
  CHECK_INPUT(cloudmap);
  const int height = cloudmap.size(0);
  const int width = cloudmap.size(1);
  T_shape Msize;
  Msize.push_back(height-30);
  Msize.push_back(width-30);
  const int M = Msize[0]*Msize[1];

  // auto options = torch::TensorOptions().dtype(torch::kInt).device(cloudmap.device());
  // cout<<cloudmap.device()<<endl;
  torch::Tensor center_map = torch::zeros({cloudmap.size(0),cloudmap.size(1)},cloudmap.options());//valid centers perserved by cloud coor num
  torch::Tensor finalcenter = torch::zeros({cloudmap.size(0),cloudmap.size(1)},cloudmap.options());//valid centers save (0,1) for removed or remained
  torch::Tensor finalcloud = torch::zeros({cloudmap.size(0),cloudmap.size(1)},cloudmap.options());

  // torch::Tensor testmap = torch::zeros({M},torch::dtype(torch::kFloat));
  // auto testmap_ptr = testmap.data_ptr<float>();

  auto centermap_ptr = center_map.data_ptr<float>();
  auto cloudmap_ptr = cloudmap.data_ptr<float>();
  auto finalcenter_ptr = finalcenter.data_ptr<float>();
  auto finalcloud_ptr = finalcloud.data_ptr<float>();
  // unsigned int *grossmapcpu;
  // grossmapcpu = (unsigned int*)malloc(M*sizeof(unsigned int));

  /*1.create centers map*/
  center_map_creator(
    cloudmap_ptr,
    centermap_ptr,
    M,
    patch,
    width,
    Msize,
    num_th);

  /*2.Dealing with equivalent problem*/
  
  equivalent(
    // grossmapcpu,
    centermap_ptr,
    cloudmap_ptr,
    finalcenter_ptr,
    // testmap_ptr,
    M,
    width,
    Msize
  );

  /*3.use my pyramid structure leaches out valid centers*/
  pyramid(
    cloudmap_ptr,
    centermap_ptr,
    finalcenter_ptr,
    width,
    M,
    Msize,
    patch,
    pylayer,
    num_ignore,
    amp);

  /*4.create finalcloudmap to find boxs coordinate for each valid centers*/
  finalcloud_creator(
    cloudmap_ptr,
    finalcloud_ptr,
    finalcenter_ptr,
    width,
    M,
    Msize,
    patch
  );
  return {center_map,finalcenter,finalcloud};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
  m.def("selecpy",&selecpy,"this is a cuda mode to leach out centers coor from variety clouds");
}











