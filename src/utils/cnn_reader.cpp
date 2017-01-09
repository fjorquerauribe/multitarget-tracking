#include "cnn_reader.hpp"

CNNReader::CNNReader(){}

CNNReader::CNNReader(string _firstCNNFeaturesFile, string _firstPreDetectionFile){
	this->CNNfeaturesFileName = _firstCNNFeaturesFile;
	this->preDetectionsFileName = _firstPreDetectionFile;
	
	/*for (int i = 0; i < 189; ++i)
	{
		MatrixXd CNNfeatures = getFeatureValues();
		vector<Rect> preDetections = getPreDetections();
		VectorXd detectionWeights = getDetectionWeights();
		cout << "preDetections size: " << preDetections.size() << "\t detectionWeights size: " << detectionWeights.size() << endl;

	}*/

}

MatrixXd CNNReader::getFeatureValues(){
	//cout << "cnn filename: " << this->CNNfeaturesFileName << endl;
	readCNNfeatures();
	return this->CNNfeatures;
}

vector<Rect> CNNReader::getPreDetections(){
	readPreDetections();
	return this->preDetections;
}

VectorXd CNNReader::getDetectionWeights(){
	return this->detectionWeights;
}

void CNNReader::readCNNfeatures(){
	std::ifstream indata;
	indata.open(this->CNNfeaturesFileName);
    
	this->CNNfeatures.resize(0,0);
	
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
        
        VectorXd rowFeatures = VectorXd::Map(values.data(), values.size());
        this->CNNfeatures.conservativeResize(this->CNNfeatures.rows() + 1, rowFeatures.size());
        this->CNNfeatures.row(this->CNNfeatures.rows() - 1) = rowFeatures;
        values.clear();
    }

    getNextCNNfeaturesFileName();
}

void CNNReader::readPreDetections(){
	std::ifstream indata;
	indata.open(this->preDetectionsFileName);
    
	this->detectionWeights.resize(0);
	this->preDetections.clear();

    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
        Rect bbox;
        bbox.x = values.at(0); bbox.y = values.at(1);
        bbox.width = values.at(2); bbox.height = values.at(3);
        this->preDetections.push_back(bbox);
        this->detectionWeights.conservativeResize(this->detectionWeights.size() + 1);
        this->detectionWeights(this->detectionWeights.size() - 1) = values.at(4);
        
        values.clear();
    }

    getNextPreDetectionsFileName();
}

void CNNReader::getNextCNNfeaturesFileName(){
	size_t index = this->CNNfeaturesFileName.find_last_of("-");
	if(index == string::npos){
		index = this->CNNfeaturesFileName.find_last_of("\\");
	}
	size_t index2 = this->CNNfeaturesFileName.find_last_of(".");
	string prefix = this->CNNfeaturesFileName.substr(0, index + 1);
	string suffix = this->CNNfeaturesFileName.substr(index2);
	string cnnFeaturesNumberString = this->CNNfeaturesFileName.substr(index + 1, index2 - index - 1);
	istringstream iss(cnnFeaturesNumberString);
	int cnnNumber = 0;
	iss >> cnnNumber;
	ostringstream oss;
	oss << (cnnNumber + 1);

	string nextCNNNumberString = oss.str();
	string nextCNNFileName = prefix + nextCNNNumberString + suffix;
	this->CNNfeaturesFileName.assign(nextCNNFileName);
}

void CNNReader::getNextPreDetectionsFileName(){
	size_t index = this->preDetectionsFileName.find_last_of("-");
	if(index == string::npos){
		index = this->preDetectionsFileName.find_last_of("\\");
	}
	size_t index2 = this->preDetectionsFileName.find_last_of(".");
	string prefix = this->preDetectionsFileName.substr(0, index + 1);
	string suffix = this->preDetectionsFileName.substr(index2);
	string preDetectionsFeaturesNumberString = this->preDetectionsFileName.substr(index + 1, index2 - index - 1);
	istringstream iss(preDetectionsFeaturesNumberString);
	int preDetectionsNumber = 0;
	iss >> preDetectionsNumber;
	ostringstream oss;
	oss << (preDetectionsNumber + 1);

	string nextPreDetectionsNumberString = oss.str();
	string nextPreDetectionsFileName = prefix + nextPreDetectionsNumberString + suffix;
	this->preDetectionsFileName.assign(nextPreDetectionsFileName);
}

/*int main(int argc, char const *argv[])
{
	string _firstCNNFeaturesFile,_firstPreDetectionFile;
	if(argc != 5)
	{
		cout << "Incorrect input list" << endl;
		cout << "exiting..." << endl;
		return EXIT_FAILURE;
	}
	else
	{
	  	if(strcmp(argv[1], "-cnn") == 0)
	  	{
	    	_firstCNNFeaturesFile = argv[2];
	  	}
	  	else
	  	{
	  		cout << "No images given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	  	if(strcmp(argv[3], "-pd") == 0)
	  	{
	    	_firstPreDetectionFile = argv[4];
	  	}
	  	else
	  	{
	  		cout << "No ground truth given" << endl;
	  		cout << "exiting..." << endl;
	  		return EXIT_FAILURE;
	  	}
	}
	CNNReader cnnReader(_firstCNNFeaturesFile, _firstPreDetectionFile);
}*/