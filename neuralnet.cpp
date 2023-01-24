#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include <sstream>

using namespace std;

class trajectoryprediction
{
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
