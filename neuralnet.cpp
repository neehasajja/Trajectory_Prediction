//here we implement the methods  themselves
#include "neuralnet.hpp"

void data::set_feature_vector(std::vector<int8_t> *);
void data::append_to_feature_vector(unit8_t);
void data::set_label();
void data::set_enumerated_label();

std::vector<int8_t> * data::get_featre_vector();
