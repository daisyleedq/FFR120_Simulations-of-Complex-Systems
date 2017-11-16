#include "layer_bank.h"

// input layer
input_layer::input_layer(){
	batch_size = 0;
    label = new Mat();
}
input_layer::~input_layer(){}

void input_layer::init_config(string namestr, int batchsize, string outputformat){
    layer_type = "input";
    layer_name = namestr;
    batch_size = batchsize;
    output_format = outputformat;
    label -> setSize(1, batch_size, 1);
}

void input_layer::forwardPass(int nsamples, const std::vector<cpuMat*>& input_data, const cpuMat* input_label){
    getSample(input_data, output_vector, input_label, label);
}

void input_layer::getSample(const std::vector<cpuMat*>& src1, std::vector<std::vector<Mat*> >& dst1, const cpuMat* src2, Mat* dst2){

	releaseVector(dst1);
	dst1.clear();
	dst2->zeros();
	// if is gradient checking, just get the 1st and 2nd sample data
    if(is_gradient_checking){
        for(int i = 0; i < 0 + batch_size; i++){
        	std::vector<Mat*> tmp;
        	Mat *tmpmat = new Mat();
        	src1[i] -> copyTo(*tmpmat);
            tmp.push_back(tmpmat);
            dst1.push_back(tmp);
            dst2 -> set(0, i, 0, src2 -> get(0, i, 0));
            tmp.clear();
            std::vector<Mat*>().swap(tmp);
        }
        return;
    }
    if(src1.size() < batch_size){
        for(int i = 0; i < src1.size(); i++){
        	std::vector<Mat*> tmp;
        	Mat *tmpmat = new Mat();
        	src1[i] -> copyTo(*tmpmat);
            tmp.push_back(tmpmat);
            dst1.push_back(tmp);
            dst2 -> set(0, i, 0, src2 -> get(0, i, 0));
            tmp.clear();
            std::vector<Mat*>().swap(tmp);
        }
        return;
    }
    std::vector<int> sample_vec;
    for(int i = 0; i < src1.size(); i++){
        sample_vec.push_back(i);
    }
    random_shuffle(sample_vec.begin(), sample_vec.end());
    for(int i = 0; i < batch_size; i++){
    	std::vector<Mat*> tmp;
    	Mat *tmpmat = new Mat();
    	src1[sample_vec[i]] -> copyTo(*tmpmat);
        tmp.push_back(tmpmat);
        dst1.push_back(tmp);
        dst2 -> set(0, i, 0, src2 -> get(0, sample_vec[i], 0));
        tmp.clear();
        std::vector<Mat*>().swap(tmp);
    }
}

void input_layer::forwardPassTest(int nsamples, const std::vector<cpuMat*>& input_data, const cpuMat* input_label){

	releaseVector(output_vector);
    output_vector.resize(input_data.size());
    label -> zeros();
    for(int i = 0; i < output_vector.size(); i++){
        output_vector[i].resize(1);
    	output_vector[i][0] = new Mat();
        input_data[i] -> copyTo(*(output_vector[i][0]));
    }
    input_label -> copyTo(*label);
}

void input_layer::backwardPass(){
    ;
}
