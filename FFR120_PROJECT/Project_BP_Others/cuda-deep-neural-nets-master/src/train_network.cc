#include "train_network.h"

using namespace std;

// FORWARD PASS INIT
// call this function when finish building network using config file,
// this function calculates the size of weights and initializes weights
// in CONV/FC/SM layers.
void forwardPassInit(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){
    //cout<<"---------------- forward init"<<endl;
    // forward pass
    int batch_size = 0;
    for(int i = 0; i < flow.size(); i++){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            batch_size = ((input_layer*)flow[i]) -> batch_size;
            ((input_layer*)flow[i]) -> forwardPass(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((convolutional_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((fully_connected_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((softmax_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }
    }
}

// FORWARD PASS
// do the forward pass using a network, this function also calculates
// the network cost from output layer.
void forwardPass(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){
    //cout<<"---------------- forward "<<endl;
    // forward pass
    int batch_size = 0;
    Mat *tmp = new Mat();
	vector3f *tmpvec3 = new vector3f();
    float J1 = 0, J2 = 0, J3 = 0, J4 = 0;
    for(int i = 0; i < flow.size(); i++){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            batch_size = ((input_layer*)flow[i]) -> batch_size;
            ((input_layer*)flow[i]) -> forwardPass(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            Mat *tmp = new Mat();
        	vector3f *tmpvec3 = new vector3f();
            for(int k = 0; k < ((convolutional_layer*)flow[i]) -> kernels.size(); ++k){
            	safeGetPt(tmp, square(((convolutional_layer*)flow[i]) -> kernels[k] -> w));
            	safeGetPt(tmpvec3, sum(tmp));
                J4 += sum(tmpvec3) * ((convolutional_layer*)flow[i]) -> kernels[k] -> weight_decay / 2.0;
            }
            tmp -> release();
            tmpvec3 -> release();
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            Mat *tmp = new Mat();
        	vector3f *tmpvec3 = new vector3f();
            safeGetPt(tmp, square(((fully_connected_layer*)flow[i]) -> w));
        	safeGetPt(tmpvec3, sum(tmp));
            J3 += sum(tmpvec3) * ((fully_connected_layer*)flow[i]) -> weight_decay / 2.0;
            tmp -> release();
            tmpvec3 -> release();
        }elif(flow[i] -> layer_type == "softmax"){
        	((softmax_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            Mat *tmp = new Mat();
        	vector3f *tmpvec3 = new vector3f();
            Mat *groundTruth = new Mat(((softmax_layer*)flow[i]) -> output_size, batch_size, 1);
            for(int i = 0; i < batch_size; i++){
            	groundTruth -> set(((input_layer*)flow[0]) -> label -> get(0, i, 0), i, 0, 1.0);
            }
            safeGetPt(tmp, log(flow[i] -> output_matrix));
            safeGetPt(tmp, multiply_elem(tmp, groundTruth));
        	safeGetPt(tmpvec3, sum(tmp));
            J1 = -sum(tmpvec3) / batch_size;
            safeGetPt(tmp, square(((softmax_layer*)flow[i]) -> w));
        	safeGetPt(tmpvec3, sum(tmp));
            J2 += sum(tmpvec3) * ((softmax_layer*)flow[i]) -> weight_decay / 2.0;
            groundTruth -> release();
            tmp -> release();
            tmpvec3 -> release();
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }

//        if(flow[i] -> output_format == "matrix"){
//        	flow[i] -> output_matrix -> printHost("OUTPUT MATRIX");
//        }else{
//        	flow[i] -> output_vector[0][0] -> printHost("OUTPUT VECTOR 00");
//          cout<<"output dimension is "<<flow[i] -> output_vector.size()<<" * "<<flow[i] -> output_vector[0].size()<<" * "<<flow[i] -> output_vector[0][0].size()<<endl;
//        }
    }
    // write network cost into network layer
    ((softmax_layer*)flow[flow.size() - 1]) -> network_cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking)
    	cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<((softmax_layer*)flow[flow.size() - 1]) -> network_cost;//endl;
}

// FORWARD PASS TEST
// This function works similar with "forwardPass", however during testing,
// some of the calculation can be simplified, for example, dropout.
void forwardPassTest(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){

    //cout<<"---------------- test "<<endl;
    // forward pass
    int batch_size = x.size();
    for(int i = 0; i < flow.size(); i++){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            ((input_layer*)flow[i]) -> forwardPassTest(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }
    }
}

// BACKWARD PASS
// do the backward pass using a network. Calculates gradients and derivatives and writes
// into network.
void backwardPass(std::vector<network_layer*> &flow){
    //cout<<"---------------- backward"<<endl;
    // backward pass
    int batch_size = ((input_layer*)flow[0]) -> batch_size;
    for(int i = flow.size() - 1; i >= 0; --i){
        //cout<<flow[i] -> layer_name<<endl;
        //cout<<" --- using gpu memory "<<double(MemoryMonitor::instance() -> getGpuMemory()) / Mb<<" Mb   before"<<endl;
        if(flow[i] -> layer_type == "input"){
            ((input_layer*)flow[i]) -> backwardPass();
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            Mat *groundTruth = new Mat(((softmax_layer*)flow[flow.size() - 1]) -> output_size, batch_size, 1);
            for(int i = 0; i < batch_size; i++){
            	groundTruth -> set(((input_layer*)flow[0]) -> label -> get(0, i, 0), i, 0, 1.0);
            }
            ((softmax_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], groundTruth);
            groundTruth -> release();
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }
        //cout<<" --- using gpu memory "<<double(MemoryMonitor::instance() -> getGpuMemory()) / Mb<<" Mb   after"<<endl;
    }
}

// UPDATE NETWORK
// Use during training, updates weights in network.
void updateNetwork(std::vector<network_layer*> &flow, int iter){
    //cout<<"---------------- update"<<endl;
    for(int i = 0; i < flow.size(); ++i){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> update(iter);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> update(iter);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> update(iter);
        }
    }
}

// PRINT NETWORK
// Print network info.
void printNetwork(std::vector<network_layer*> &flow){
    cout<<"****************************************************************************"<<endl
        <<"**                       NETWORK LAYERS                                     "<<endl
        <<"****************************************************************************"<<endl<<endl;
    for(int i = 0; i < flow.size(); ++i){
        cout<<"##-------------------layer number "<<i<<", layer name is "<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            cout<<"batch size = "<<((input_layer*)flow[i]) -> batch_size<<endl;
        }elif(flow[i] -> layer_type == "convolutional"){
            cout<<"kernel amount = "<<((convolutional_layer*)flow[i]) -> kernels.size()<<endl;
            cout<<"kernel size = ["<<((convolutional_layer*)flow[i]) -> kernels[0] -> w -> rows<<", "<<((convolutional_layer*)flow[i]) -> kernels[0] -> w -> cols<<"]"<<endl;
            cout<<"padding = "<<((convolutional_layer*)flow[i]) -> padding<<endl;
            cout<<"stride = "<<((convolutional_layer*)flow[i]) -> stride<<endl;
            cout<<"combine feature map = "<<((convolutional_layer*)flow[i]) -> combine_feature_map<<endl;
            cout<<"weight decay = "<<((convolutional_layer*)flow[i]) -> kernels[0] -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "fully_connected"){
            cout<<"hidden size = "<<((fully_connected_layer*)flow[i]) -> size<<endl;
            cout<<"weight decay = "<<((fully_connected_layer*)flow[i]) -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "softmax"){
            cout<<"output size = "<<((softmax_layer*)flow[i]) -> output_size<<endl;
            cout<<"weight decay = "<<((softmax_layer*)flow[i]) -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "combine"){
            ;
        }elif(flow[i] -> layer_type == "branch"){
            ;
        }elif(flow[i] -> layer_type == "non_linearity"){
            cout<<"non-lin method = "<<((non_linearity_layer*)flow[i]) -> method<<endl;
        }elif(flow[i] -> layer_type == "pooling"){
            cout<<"pooling method = "<<((pooling_layer*)flow[i]) -> method<<endl;
            cout<<"overlap = "<<((pooling_layer*)flow[i]) -> overlap<<endl;
            cout<<"stride = "<<((pooling_layer*)flow[i]) -> stride<<endl;
            cout<<"window size = "<<((pooling_layer*)flow[i]) -> window_size<<endl;
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            cout<<"alpha = "<<((local_response_normalization_layer*)flow[i]) -> alpha<<endl;
            cout<<"beta = "<<((local_response_normalization_layer*)flow[i]) -> beta<<endl;
            cout<<"k = "<<((local_response_normalization_layer*)flow[i]) -> k<<endl;
            cout<<"n = "<<((local_response_normalization_layer*)flow[i]) -> n<<endl;
        }elif(flow[i] -> layer_type == "dropout"){
            cout<<"dropout rate = "<<((dropout_layer*)flow[i]) -> dropout_rate<<endl;
        }
        if(flow[i] -> output_format == "matrix"){
            cout<<"output matrix size is ["<<flow[i] -> output_matrix -> rows<<", "<<flow[i] -> output_matrix -> cols<<", "<<flow[i] -> output_matrix -> channels<<"]"<<endl;
        }else{
            cout<<"output vector size is "<<flow[i] -> output_vector.size()<<" * "<<flow[i] -> output_vector[0].size()<<" * ["<<flow[i] -> output_vector[0][0] -> rows<<", "<<flow[i] -> output_vector[0][0] -> cols<<", "<<flow[i] -> output_vector[0][0] -> channels<<"]"<<endl;
        }
        cout<<"---------------------"<<endl;
    }
}

// TEST NETWORK
// Get testing data, and feed into the trained network, compare result with testing label,
// then calculates the accuracy.
void testNetwork(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){

    int batch_size = 100;
    //int batch_amount = 5;
    int batch_amount = x.size() / batch_size;
    int correct = 0;
    Mat *tmp = new Mat();
    Mat *tmp2 = new Mat();
    for(int i = 0; i < batch_amount; i++){
    	cout<<"processing batch number "<<i<<endl;
        std::vector<cpuMat*> batchX;
        cpuMat* batchY = new cpuMat(1, batch_size, 1);
        for(int j = 0; j < batch_size; j++){
        	cpuMat *tmpmat = new cpuMat();
        	x[i * batch_size + j] -> copyTo(*tmpmat);
            batchX.push_back(tmpmat);
            batchY -> set(0, j, 0, y -> get(0, i * batch_size + j, 0));
        }
        forwardPassTest(batchX, batchY, flow);
        safeGetPt(tmp, findMax(flow[flow.size() - 1] -> output_matrix));
        batchY -> copyTo(*tmp2);
        correct += sameValuesInMat(tmp, tmp2);
        batchY -> release();
        releaseVector(batchX);
        batchX.clear();
        std::vector<cpuMat*>().swap(batchX);
        cout<<" --- using gpu memory "<<MemoryMonitor::instance() -> getGpuMemory() / Mb<<" Mb"<<endl;
    }
    if(x.size() % batch_size){
        std::vector<cpuMat*> batchX;
        cpuMat* batchY = new cpuMat(1, x.size() % batch_size, 1);
        for(int j = 0; j < x.size() % batch_size; j++){
        	cpuMat *tmpmat = new cpuMat();
        	x[batch_amount * batch_size + j] -> copyTo(*tmpmat);
            batchX.push_back(tmpmat);
            batchY -> set(0, j, 0, y -> get(0, batch_amount * batch_size + j, 0));
        }
        forwardPassTest(batchX, batchY, flow);
        safeGetPt(tmp, findMax(flow[flow.size() - 1] -> output_matrix));
        batchY -> copyTo(*tmp2);
        correct += sameValuesInMat(tmp, tmp2);
        batchY -> release();
        releaseVector(batchX);
        batchX.clear();
        std::vector<cpuMat*>().swap(batchX);
    }
    tmp -> release();
    tmp2 -> release();
    cout<<"correct: "<<correct<<", total: "<<x.size()<<", accuracy: "<<float(correct) / (float)(x.size())<<endl;
}

// TRAIN NETWORK
// train network, for each training epoch, there should be a forward pass, a
// backward pass, and a update.
void trainNetwork(const std::vector<cpuMat*> &x, const cpuMat *y, const std::vector<cpuMat*> &tx, const cpuMat *ty, std::vector<network_layer*> &flow){

    if (is_gradient_checking){
        gradient_checking_network_layers(flow, x, y);
    }else{
    	string path;
    	if(use_log){
			path = "saved_data";
			mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			saveNetworkConfig(path + "/saved_config.txt", flow);
    	}
		cout<<"****************************************************************************"<<endl
			<<"**                       TRAINING NETWORK......                             "<<endl
			<<"****************************************************************************"<<endl<<endl;
        int k = 0;
        for(int epo = 1; epo <= training_epochs; epo++){
            for(; k <= iter_per_epo * epo; k++){
                cout<<"epoch: "<<epo<<", iter: "<<k; // <<endl;
                forwardPass(x, y, flow);
                backwardPass(flow);
                updateNetwork(flow, k);
                cout<<" --- using gpu memory "<<MemoryMonitor::instance() -> getGpuMemory() / Mb<<" Mb"<<endl;
            }
            //cout<<"Test training data: "<<endl;
            //testNetwork(x, y, flow);
            cout<<"Test testing data: "<<endl;
            testNetwork(tx, ty, flow);
        	if(use_log){
                string path_iter = path + "/iter_" + std::to_string(k);
                saveNetwork(path_iter, flow);
        	}
        }
    }
}


