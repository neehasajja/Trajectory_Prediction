#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
// build model resnet50
struct BasicBlock : torch::nn::Module {

static const int expansion;

int 64_t stride;
torch::nn::Conv2d conv1;
torch::nn::BatchNorm bn1;
torch::nn::Conv2d conv2,;
torch::nn::BatchNorm bn2;
torch::nn::Sequential downsample;
}


BasicBlock(int64_t inplanes, int64_t planes,int64_t, int64_t stride_=1,
  
