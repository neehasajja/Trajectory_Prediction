//this is the data handler set which implements all the logic needed to read in the data split data count the no.of unique classes convert t little endian and pass around the train data ,test data and validation data //
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unorderedset>



class trajectoryprediction
{
   std::vector<data *> *data_array;
   std::vector<data *> *training_data;
   std::vector<data *> *test_data;
   std::vector<data *> *val_data;

public:
  (const vector<unsigned> &topology);
  void feedForward(const vector<double> &inputVals);
  void backProp(const vector<double> &targetVals);
  void getResults(vector<double> &resultVals) const;

private:
};

// set environment variable for data
     const char *path="/home/neehasajja/file.txt";
     std::ofstream file(path); //open in constructor
     std::string data("data to wrtite to file");
     file << data;
     std::filesystem::path -


// get config
const char config[] = "url=http://example.com\n"
                  "file=main.exe\n"
                  "true=0";

std::istringstream is_file(config);

std::string line;
while( std::getline(is_file, line) )
{
std::istringstream is_line(line);
std::string key;
if( std::getline(is_line, key, '=') )
{
std::string value;
if( std::getline(is_line, value) )
  store_line(key, value);
}
}
