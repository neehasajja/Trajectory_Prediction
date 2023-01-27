#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "layer.hpp"
#include "data.h"

class OutputLayer : public Layer
{
  public:
  OutputLayer(int prev, int current) : Layer(prev, curr){}
  void feedForward(Layer);
  void backProp(data *data); // here we have to mnetion what kind of class we want as output
  void updateWeights(double, Layer *);
}

#endif
