//this is the data handler set which implements all the logic needed to read in the data split data count the no.of unique classes convert t little endian and pass around the train data ,test data and validation data //
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <unorderedset>
#include <string>


class trajectoryprediction
{
   std::vector<data *> *data_array; //all of the data
   std::vector<data *> *training_data;
   std::vector<data *> *test_data;
   std::vector<data *> *validation_data;

   int num_classes;
   int feature_vector_size;
   std::map<int8_t, int> class_map;

   const double TRAIN_SET_PERCENT = 0.75;
   const double TEST_SET_PERCENT = 0.25;
   const double VALIDATE_SET_PERCENT = 0.25;

   public:
   data_handler();
   ~dat_handler();

   void read_feature_vector(std::string path);
   void read_feature_vector(std::labels path);
   void split_data();
   void count_classes();

   uint32_t convert_to_little_endian(const unsigned char* bytes);
   std::vector *> get_trtaining_data();
   std::vector *> get_test_data();
   std::vector *> get_validation_data();

};

#endif
