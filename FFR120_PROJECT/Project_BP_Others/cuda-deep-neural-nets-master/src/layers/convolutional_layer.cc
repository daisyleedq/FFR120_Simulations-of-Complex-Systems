#include "layer_bank.h"

// convolutional kernel
convolutional_kernel::convolutional_kernel(){
	weight_decay = 0.0;
	kernel_size = 0;
    w = NULL;
    b = NULL;
    wgrad = NULL;
    bgrad = NULL;
    wd2 = NULL;
    bd2 = NULL;
}
convolutional_kernel::~convolutional_kernel(){
	w -> release();
	wgrad -> release();
    wd2 -> release();
    b -> release();
    bgrad -> release();
    bd2 -> release();
}

void convolutional_kernel::init_config(int width, float weightDecay){
    float epsilon = 0.12;
    kernel_size = width;
    weight_decay = weightDecay;
    w = new Mat(kernel_size, kernel_size, 3);
    w -> randn();
    safeGetPt(w, multiply_elem(w, epsilon));
    wgrad = new Mat(kernel_size, kernel_size, 3);
    wd2 = new Mat(kernel_size, kernel_size, 3);
    b = new vector3f();
    bgrad = new vector3f();
    bd2 = new vector3f();
}

void convolutional_kernel::release(){
	w -> release();
	wgrad -> release();
    wd2 -> release();
    b -> release();
    bgrad -> release();
    bd2 -> release();
}

// convolutional layer
convolutional_layer::convolutional_layer(){
	stride = 1;
	padding = 0;
	combine_feature_map = 0;
    momentum_derivative = 0.0;
    momentum_second_derivative = 0.0;
    iter = 0;
    mu = 0.0;
    combine_weight = NULL;
    combine_weight_grad = NULL;
    combine_weight_d2 = NULL;
    velocity_combine_weight = NULL;
    second_derivative_combine_weight = NULL;
    learning_rate_w = NULL;
    learning_rate_b = NULL;

}
convolutional_layer::~convolutional_layer(){

	for(int i = 0; i < kernels.size(); ++i){
		kernels[i] -> release();
	}
    kernels.clear();
    vector<convolutional_kernel*>().swap(kernels);
    combine_weight -> release();
    combine_weight_grad -> release();
    combine_weight_d2 -> release();
    velocity_combine_weight -> release();
    second_derivative_combine_weight -> release();
    learning_rate_w -> release();
    learning_rate_b -> release();
}

void convolutional_layer::init_config(string namestr, int kernel_amount, int kernel_size, int output_amount, int _padding, int _stride, float weight_decay, string outputformat){

    layer_type = "convolutional";
    layer_name = namestr;
    output_format = outputformat;
    padding = _padding;
    stride = _stride;
    combine_feature_map = output_amount;
    kernels.clear();
    kernels.resize(kernel_amount);
    for(int i = 0; i < kernels.size(); ++i){
    	kernels[i] = new convolutional_kernel();
    	kernels[i] -> init_config(kernel_size, weight_decay);
    }
}

void convolutional_layer::init_weight(network_layer* previous_layer){
    float epsilon = 0.12;
    if(combine_feature_map > 0){
    	combine_weight = new Mat(kernels.size(), combine_feature_map, 1);
    	combine_weight -> randn();
        safeGetPt(combine_weight, multiply_elem(combine_weight, epsilon));
        combine_weight_grad = new Mat(kernels.size(), combine_feature_map, 1);
        combine_weight_d2 = new Mat(kernels.size(), combine_feature_map, 1);
        velocity_combine_weight = new Mat(kernels.size(), combine_feature_map, 1);
        second_derivative_combine_weight = new Mat(kernels.size(), combine_feature_map, 1);
    }
    // updater
    learning_rate_w = new Mat(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    learning_rate_b = new vector3f();
    velocity_w.resize(kernels.size());
    velocity_b.resize(kernels.size());
    second_derivative_w.resize(kernels.size());
    second_derivative_b.resize(kernels.size());
    for(int i = 0; i < kernels.size(); ++i){
    	velocity_w[i] = new Mat(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	second_derivative_w[i] = new Mat(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	velocity_b[i] = new vector3f();
    	second_derivative_b[i] = new vector3f();
    }
    iter = 0;
    mu = 1e-2;
    convolutional_layer::setMomentum();
}

void convolutional_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void convolutional_layer::update(int iter_num){

    iter = iter_num;
    if(iter == 30) convolutional_layer::setMomentum();

    vector3f *allmu = new vector3f(mu, mu, mu);
    Mat *tmp = new Mat();
    vector3f *tmpvec = new vector3f();
    for(int i = 0; i < kernels.size(); ++i){
        safeGetPt(second_derivative_w[i], multiply_elem(second_derivative_w[i], momentum_second_derivative));
        safeGetPt(tmp, multiply_elem(kernels[i] -> wd2, 1.0 - momentum_second_derivative));
        safeGetPt(second_derivative_w[i], add(second_derivative_w[i], tmp));
        safeGetPt(tmp, add(second_derivative_w[i], mu));
        safeGetPt(learning_rate_w, divide(lrate_w, tmp));
        safeGetPt(velocity_w[i], multiply_elem(velocity_w[i], momentum_derivative));
        safeGetPt(tmp, multiply_elem(kernels[i] -> wgrad, learning_rate_w));
        safeGetPt(tmp, multiply_elem(tmp, 1.0 - momentum_derivative));
        safeGetPt(velocity_w[i], add(tmp, velocity_w[i]));
        safeGetPt(kernels[i] -> w, subtract(kernels[i] -> w, velocity_w[i]));

        safeGetPt(second_derivative_b[i], multiply_elem(second_derivative_b[i], momentum_second_derivative));
        safeGetPt(tmpvec, multiply_elem(kernels[i] -> bd2, 1.0 - momentum_second_derivative));
        safeGetPt(second_derivative_b[i], add(second_derivative_b[i], tmpvec));
        safeGetPt(tmpvec, add(second_derivative_b[i], mu));
        safeGetPt(learning_rate_b, divide(lrate_b, tmpvec));
        safeGetPt(velocity_b[i], multiply_elem(velocity_b[i], momentum_derivative));
        safeGetPt(tmpvec, multiply_elem(kernels[i] -> bgrad, learning_rate_b));
        safeGetPt(tmpvec, multiply_elem(tmpvec, 1.0 - momentum_derivative));
        safeGetPt(velocity_b[i], add(tmpvec, velocity_b[i]));
        safeGetPt(kernels[i] -> b, subtract(kernels[i] -> b, velocity_b[i]));
    }
    if(combine_feature_map > 0){
        safeGetPt(second_derivative_combine_weight, multiply_elem(second_derivative_combine_weight, momentum_second_derivative));
        safeGetPt(tmp, multiply_elem(combine_weight_d2, 1.0 - momentum_second_derivative));
        safeGetPt(second_derivative_combine_weight, add(second_derivative_combine_weight, tmp));
        safeGetPt(tmp, add(second_derivative_combine_weight, mu));
        safeGetPt(learning_rate_w, divide(lrate_w, tmp));
        safeGetPt(velocity_combine_weight, multiply_elem(velocity_combine_weight, momentum_derivative));
        safeGetPt(tmp, multiply_elem(combine_weight_grad, learning_rate_w));
        safeGetPt(tmp, multiply_elem(tmp, 1.0 - momentum_derivative));
        safeGetPt(velocity_combine_weight, add(tmp, velocity_combine_weight));
        safeGetPt(combine_weight, subtract(combine_weight, velocity_combine_weight));
    }
    tmp -> release();
    tmpvec -> release();
}

void convolutional_layer::forwardPass(int nsamples, network_layer* previous_layer){

    std::vector<std::vector<Mat*> > input;
    if(previous_layer -> output_format == "image"){
    	copyVector(previous_layer -> output_vector, input);
    }else{
        // no!!!!
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }

    Mat *c_weight = new Mat();
    Mat *tmp = new Mat();
	vector3f *tmpvec3 = new vector3f();
    if(combine_feature_map > 0){
    	safeGetPt(c_weight, exp(combine_weight));
    	safeGetPt(tmp, reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
    	safeGetPt(tmp, repmat(tmp, c_weight -> rows, 1));
    	safeGetPt(c_weight, divide(c_weight, tmp));
    }
    releaseVector(output_vector);
    output_vector.clear();
    output_vector.resize(input.size());
    for(int i = 0; i < output_vector.size(); ++i){
    	output_vector[i].clear();
        if(combine_feature_map > 0){output_vector[i].resize(combine_feature_map * input[i].size());}
        else{ 						output_vector[i].resize(kernels.size() * input[i].size()); }
        for(int j = 0; j < output_vector[i].size(); ++j){
        	output_vector[i][j] = new Mat();
        }
    }
    std::vector<Mat*> tmpvec(kernels.size(), NULL);

    for(int i = 0; i < input.size(); ++i){
        for(int j = 0; j < input[i].size(); ++j){
            for(int k = 0; k < kernels.size(); ++k){
                safeGetPt(tmp, rot90(kernels[k] -> w, 2));
                safeGetPt(tmpvec[k], conv2(input[i][j], tmp, CONV_VALID, padding, stride));
                safeGetPt(tmpvec[k], add(tmpvec[k], kernels[k] -> b));
            }
            if(combine_feature_map > 0){
                std::vector<Mat*> outputvec(combine_feature_map, NULL);
                for(int k = 0; k < outputvec.size(); k++) {
                	outputvec[k] = new Mat(tmpvec[0] -> rows, tmpvec[0] -> cols, 3);
                }
                for(int m = 0; m < kernels.size(); m++){
                    for(int n = 0; n < combine_feature_map; n++){
                    	float tmpfloat = c_weight -> get(m, n, 0);
                        safeGetPt(tmp, multiply_elem(tmpvec[m], tmpfloat));
                        safeGetPt(outputvec[n], add(outputvec[n], tmp));
                    }
                }
                for(int k = 0; k < outputvec.size(); k++) {
                	outputvec[k] -> copyTo(*(output_vector[i][k + j * combine_feature_map]));
                }
                releaseVector(outputvec);
                outputvec.clear();
                std::vector<Mat*>().swap(outputvec);
            }else{
                for(int k = 0; k < kernels.size(); k++) {
                	tmpvec[k] -> copyTo(*(output_vector[i][k + j * kernels.size()]));
                }
            }
        }
    }
    releaseVector(tmpvec);
    tmpvec.clear();
    std::vector<Mat*>().swap(tmpvec);
    tmpvec3 -> release();
    c_weight -> release();
    tmp -> release();
    releaseVector(input);
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
}

void convolutional_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    convolutional_layer::forwardPass(nsamples, previous_layer);
}

void convolutional_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    std::vector<std::vector<Mat*> > derivative;
    std::vector<std::vector<Mat*> > deriv2;

    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0] -> rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0] -> rows);
    }else{
    	copyVector(next_layer -> delta_vector, derivative);
    	copyVector(next_layer -> d2_vector, deriv2);
    }

    std::vector<std::vector<Mat*> > input;
    if(previous_layer -> output_format == "image"){
    	copyVector(previous_layer -> output_vector, input);
    }else{
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }

    releaseVector(delta_vector);
    releaseVector(d2_vector);
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(input.size());
    d2_vector.resize(input.size());
    for(int i = 0; i < delta_vector.size(); i++){
    	delta_vector[i].clear();
    	d2_vector[i].clear();
        delta_vector[i].resize(input[i].size());
        d2_vector[i].resize(input[i].size());
        for(int j = 0; j < delta_vector[i].size(); ++j){
        	delta_vector[i][j] = new Mat();
        	d2_vector[i][j] = new Mat();
        }
    }
    Mat *tmp = new Mat();
    Mat *tmp2 = new Mat();
    Mat *tmp3 = new Mat();
    Mat *tmp4 = new Mat();
	vector3f *tmpvec3_1 = new vector3f();
	vector3f *tmpvec3_2 = new vector3f();
    std::vector<Mat*> tmp_wgrad(kernels.size(), NULL);
    std::vector<Mat*> tmp_wd2(kernels.size(), NULL);
    std::vector<vector3f*> tmpgradb(kernels.size(), NULL);
    std::vector<vector3f*> tmpbd2(kernels.size(), NULL);
    std::vector<Mat*> sensi(kernels.size(), NULL);
    std::vector<Mat*> sensid2(kernels.size(), NULL);
    for(int m = 0; m < kernels.size(); m++) {
    	sensi[m] = new Mat(derivative[0][0] -> rows, derivative[0][0] -> cols, 3);
    	sensid2[m] = new Mat(deriv2[0][0] -> rows, deriv2[0][0] -> cols, 3);
    	tmp_wgrad[m] = new Mat(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	tmp_wd2[m] = new Mat(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	tmpgradb[m] = new vector3f();
    	tmpbd2[m] = new vector3f();
    }
	Mat *c_weight = NULL;
    Mat *c_weightgrad = NULL;
    Mat *c_weightd2 = NULL;
    Mat *tmpinput = new Mat();

    if(combine_feature_map > 0){
        c_weight = new Mat(combine_weight -> rows, combine_weight -> cols, 1);
        c_weightgrad = new Mat(combine_weight -> rows, combine_weight -> cols, 1);
        c_weightd2 = new Mat(combine_weight -> rows, combine_weight -> cols, 1);
    	safeGetPt(c_weight, exp(combine_weight));
    	safeGetPt(tmp, reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
    	safeGetPt(tmp, repmat(tmp, c_weight -> rows, 1));
    	safeGetPt(c_weight, divide(c_weight, tmp));
    }
	Mat *tmp_delta = new Mat();
    Mat *tmp_d2 = new Mat();
    for(int i = 0; i < nsamples; i++){
        for(int j = 0; j < input[i].size(); j++){
            if(padding > 0){ safeGetPt(tmpinput, dopadding(input[i][j], padding)); }
            else{ input[i][j] -> copyTo(*tmpinput); }

            for(int m = 0; m < kernels.size(); m++) {
            	sensi[m] -> zeros();
            	sensid2[m] -> zeros();
            	if(combine_feature_map > 0){
                    for(int n = 0; n < combine_feature_map; n++){
                    	tmpvec3_1 -> setAll(c_weight -> get(m, n, 0));
                    	float tmpfloat = powf(c_weight -> get(m, n, 0), 2);
                    	tmpvec3_2 -> setAll(tmpfloat);
                    	safeGetPt(tmp, multiply_elem(derivative[i][j * combine_feature_map + n], tmpvec3_1));
                    	safeGetPt(tmp2, multiply_elem(deriv2[i][j * combine_feature_map + n], tmpvec3_2));
                    	safeGetPt(sensi[m], add(sensi[m], tmp));
                    	safeGetPt(sensid2[m], add(sensid2[m], tmp2));
                    }
                }else{
                	derivative[i][j * kernels.size() + m] -> copyTo(*(sensi[m]));
                	deriv2[i][j * kernels.size() + m] -> copyTo(*(sensid2[m]));
                }
            	if(stride > 1){
                    int len = input[0][0] -> rows + padding * 2 - kernels[0] -> w -> rows + 1;
                	safeGetPt(sensi[m], interpolation(sensi[m], len));
                	safeGetPt(sensid2[m], interpolation(sensid2[m], len));
                }

                // calculate derivative[i][j]
            	safeGetPt(tmp3, square(kernels[m] -> w));
            	safeGetPt(tmp, conv2(sensi[m], kernels[m] -> w, CONV_FULL, 0, 1));
            	safeGetPt(tmp2, conv2(sensid2[m], tmp3, CONV_FULL, 0, 1));
            	if(0 == m){
                	tmp -> copyTo(*tmp_delta);
                	tmp2 -> copyTo(*tmp_d2);
                }else{
                	safeGetPt(tmp_delta, add(tmp_delta, tmp));
                	safeGetPt(tmp_d2, add(tmp_d2, tmp2));
                }
            	// calculate kernel[m]'s gradient
                safeGetPt(tmp, square(tmpinput));
            	safeGetPt(tmp2, rot90(sensi[m], 2));
            	safeGetPt(tmp3, rot90(sensid2[m], 2));
            	safeGetPt(tmpvec3_1, sum(tmp2));
            	safeGetPt(tmpvec3_2, sum(tmp3));
            	safeGetPt(tmpgradb[m], add(tmpgradb[m], tmpvec3_1));
            	safeGetPt(tmpbd2[m], add(tmpbd2[m], tmpvec3_2));
            	safeGetPt(tmp2, conv2(tmpinput, tmp2, CONV_VALID, 0, 1));
            	safeGetPt(tmp3, conv2(tmp, tmp3, CONV_VALID, 0, 1));
            	safeGetPt(tmp_wgrad[m], add(tmp_wgrad[m], tmp2));
            	safeGetPt(tmp_wd2[m], add(tmp_wd2[m], tmp3));

                if(combine_feature_map > 0){
                    // combine feature map weight matrix (after softmax)
                	safeGetPt(tmp2, rot90(kernels[m] -> w, 2));
                	tmp2 -> copyTo(*tmp3);
                	safeGetPt(tmp4, square(tmp3));
                	safeGetPt(tmp2, conv2(tmpinput, tmp2, CONV_VALID, padding, stride));
                	safeGetPt(tmp3, conv2(tmp, tmp4, CONV_VALID, padding, stride));
                    for(int n = 0; n < combine_feature_map; n++){
                        Mat *tmpd = new Mat();
                        safeGetPt(tmpd, multiply_elem(derivative[i][j * combine_feature_map + n], tmp2));
                        c_weightgrad -> set(m, n, 0, c_weightgrad -> get(m, n, 0) + sum(tmpd) -> get(0));
                        safeGetPt(tmpd, multiply_elem(deriv2[i][j * combine_feature_map + n], tmp3));
                        c_weightd2 -> set(m, n, 0, c_weightd2 -> get(m, n, 0) + sum(tmpd) -> get(0));
                        tmpd -> release();
                    }
                }
            } // for(int m = 0; m < kernels.size(); m++) ends
            if(padding > 0){
            	safeGetPt(tmp_delta, depadding(tmp_delta, padding));
            	safeGetPt(tmp_d2, depadding(tmp_d2, padding));
            }
            tmp_delta -> copyTo(*(delta_vector[i][j]));
        	tmp_d2 -> copyTo(*(d2_vector[i][j]));
        }
    }
	tmp_delta -> release();
	tmp_d2 -> release();
    for(int i = 0; i < kernels.size(); i++){
    	safeGetPt(kernels[i] -> wgrad, divide(tmp_wgrad[i], (float)nsamples));
    	safeGetPt(tmp, multiply_elem(kernels[i] -> w, kernels[i] -> weight_decay));
    	safeGetPt(kernels[i] -> wgrad, add(tmp, kernels[i] -> wgrad));
    	safeGetPt(kernels[i] -> wd2, divide(tmp_wd2[i], (float)nsamples));
    	safeGetPt(kernels[i] -> wd2, add(kernels[i] -> wd2, kernels[i] -> weight_decay));
    	safeGetPt(kernels[i] -> bgrad, divide(tmpgradb[i], (float)nsamples));
    	safeGetPt(kernels[i] -> bd2, divide(tmpbd2[i], (float)nsamples));
    }
    if(combine_feature_map > 0){
    	safeGetPt(tmp2, multiply_elem(c_weightgrad, c_weight));
    	safeGetPt(tmp2, reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
    	safeGetPt(tmp2, repmat(tmp2, c_weightgrad -> rows, 1));
    	safeGetPt(tmp, subtract(c_weightgrad, tmp2));
    	safeGetPt(tmp, multiply_elem(c_weight, tmp));
    	safeGetPt(combine_weight_grad, divide(tmp, (float)nsamples));

    	safeGetPt(tmp2, multiply_elem(c_weightd2, c_weight));
    	safeGetPt(tmp2, reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
    	safeGetPt(tmp2, repmat(tmp2, c_weightd2 -> rows, 1));
    	safeGetPt(tmp, subtract(c_weightd2, tmp2));
    	safeGetPt(tmp, multiply_elem(c_weight, tmp));
    	safeGetPt(combine_weight_d2, divide(tmp, (float)nsamples));
    }
	// release
    tmpinput -> release();
    tmp -> release();
    tmp2 -> release();
    tmp3 -> release();
    tmp4 -> release();
    tmpvec3_1 -> release();
    tmpvec3_2 -> release();
    if(combine_feature_map > 0){
        c_weight -> release();
        c_weightgrad -> release();
        c_weightd2 -> release();
    }
    releaseVector(sensi);
    sensi.clear();
    std::vector<Mat*>().swap(sensi);
    releaseVector(sensid2);
    sensid2.clear();
    std::vector<Mat*>().swap(sensid2);
    releaseVector(tmp_wgrad);
    tmp_wgrad.clear();
    std::vector<Mat*>().swap(tmp_wgrad);
    releaseVector(tmp_wd2);
    tmp_wd2.clear();
    std::vector<Mat*>().swap(tmp_wd2);
    releaseVector(input);
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
    releaseVector(derivative);
    derivative.clear();
    std::vector<std::vector<Mat*> >().swap(derivative);
    releaseVector(deriv2);
    deriv2.clear();
    std::vector<std::vector<Mat*> >().swap(deriv2);
    releaseVector(tmpgradb);
    tmpgradb.clear();
    std::vector<vector3f*>().swap(tmpgradb);
    releaseVector(tmpbd2);
    tmpbd2.clear();
    std::vector<vector3f*>().swap(tmpbd2);
}


