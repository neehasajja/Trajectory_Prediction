#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <vector>

class  LAYER
{
    public:
    int currentLayerSize = 50;
    std::vector<Neuron *> neurons;
    std::vector<double> layerOutput;
    Layer(int, int);
    ~Layer();
    std::vector<double> getLayerOutputs();

  }

  #endif
