#include "data_structure.h"

using namespace std;

cpuMat::cpuMat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

cpuMat::cpuMat(const cpuMat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	Data = NULL;
	mallocMat();
	memcpy(Data, m.Data, getLength() * sizeof(float));
}

cpuMat::cpuMat(const Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	Data = NULL;
	mallocMat();
	checkCudaErrors(cudaMemcpy(Data, m.Data, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
}

cpuMat::cpuMat(int height, int width, int nchannels){
	cols = width;
	rows = height;
	channels = nchannels;
	Data = NULL;
	mallocMat();
}
cpuMat::~cpuMat(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
}

void cpuMat::release(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

cpuMat& cpuMat::operator=(const cpuMat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	if(NULL != Data){
		MemoryMonitor::instance()->freeCpuMemory(Data);
		Data = NULL;
	}
	mallocMat();
	memcpy(Data, m.Data, getLength() * sizeof(float));
    return *this;
}

cpuMat& cpuMat::operator<<=(cpuMat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	if(NULL != Data){
		MemoryMonitor::instance()->freeCpuMemory(Data);
		Data = NULL;
	}
	mallocMat();
	memcpy(Data, m.Data, getLength() * sizeof(float));
	m.release();
    return *this;
}

void cpuMat::setSize(int r, int c, int ch){
	rows = r;
	cols = c;
	channels = ch;
	if(NULL != Data){
		MemoryMonitor::instance()->freeCpuMemory(Data);
		Data = NULL;
	}
	mallocMat();
	zeros();
}

void cpuMat::zeros(){
	setAll(0.0);
}

void cpuMat::ones(){
	setAll(1.0);
}

void cpuMat::randn(){
	if(NULL == Data) mallocMat();
	int len = cols * rows;
	curandGenerator_t gen;
	// Create pseudo-random number generator
	checkCudaErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	for(int ch = 0; ch < channels; ++ch){
		// Set seed
		checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, rand() % 6789));
		// Generate n floats on device
		if(len % 2){
			// In general, the normal generating functions (e.g. curandGenerateNormal, curandGenerateLogNormal, etc.)
			// require the number of requested points to be a multiple of 2, for a pseudorandom RNG.
			float *tmp = NULL;
			tmp = (float*)MemoryMonitor::instance()->cpuMalloc((len + 1) * sizeof(float));
			checkCudaErrors(curandGenerateNormal(gen, tmp, len + 1, 0.0, 1.0));
			memcpy(Data + ch * len, tmp, len * sizeof(float));
			MemoryMonitor::instance()->freeCpuMemory(tmp);
		}else{
			checkCudaErrors(curandGenerateNormal(gen, Data + ch * len, len, 0.0, 1.0));
		}
	}
	// Cleanup generator
	checkCudaErrors(curandDestroyGenerator(gen));
}

void cpuMat::set(int pos_y, int pos_x, int pos_channel, float val){
	if(NULL == Data) {zeros();}
	if(pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	Data[RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols)] = val;
}

void cpuMat::set(int pos_y, int pos_x, const vector3f& val){
	if(NULL == Data ) {zeros();}
	if(pos_x >= cols || pos_y >= rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < channels; ++i){
		set(pos_y, pos_x, i, val.get(i));
	}
}

void cpuMat::set(int pos, const vector3f& val){
	if(NULL == Data ) {zeros();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < channels; ++i){
		Data[pos + i * (rows * cols)] = val.get(i);
	}
}

void cpuMat::set(int pos, int pos_channel, float val){
	if(NULL == Data ) {zeros();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	Data[pos + pos_channel * (rows * cols)] = val;
}

void cpuMat::setAll(float val){
	if(NULL == Data) {mallocMat();}
	int len = getLength();
	for(int i = 0; i < len; ++i){
		Data[i] = val;
	}
}

void cpuMat::setAll(const vector3f &v){
	if(NULL == Data) {mallocMat();}
	int len = rows * cols;
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < len; ++i){
			Data[len * ch + i] = v.get(ch);
		}
	}
}

float cpuMat::get(int pos_y, int pos_x, int pos_channel) const{
	if(NULL == Data ||
	   pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	return Data[RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols)];
}

vector3f cpuMat::get(int pos_y, int pos_x) const{
	if(NULL == Data ||
	   pos_x >= cols || pos_y >= rows || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	for(int i = 0; i < 3; ++i){
		res.set(i, Data[RC2IDX(pos_y, pos_x, cols) + i * (rows * cols)]);
	}
	return res;
}

int cpuMat::getLength() const{
	return rows * cols * channels;
}

void cpuMat::copyTo(cpuMat &m) const{
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocMat();
	memcpy(m.Data, Data, getLength() * sizeof(float));
}

void cpuMat::copyTo(Mat &m) const{
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeGpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocDevice();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyHostToDevice));
}

void cpuMat::moveTo(cpuMat &m){
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocMat();
	memcpy(m.Data, Data, getLength() * sizeof(float));
	release();
}

void cpuMat::moveTo(Mat &m){
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeGpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocDevice();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyHostToDevice));
	release();
}

// only changes devData (on GPU)
cpuMat cpuMat::operator+(const cpuMat &m) const{
	if(NULL == Data  || NULL == m.Data|| getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(m);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = tmp.Data[i] + Data[i];
	}
	return tmp;
}

cpuMat cpuMat::operator+(float val) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(rows, cols, channels);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] + val;
	}
	return tmp;
}

cpuMat cpuMat::operator+(const vector3f &v) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	cpuMat tmp(rows, cols, channels);
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			tmp.Data[i + n * ch] = Data[i + n * ch] + v.get(ch);
		}
	}
	return tmp;
}

cpuMat& cpuMat::operator+=(const cpuMat &m){
	if(NULL == Data  || NULL == m.Data|| getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	for(int i = 0; i < n; ++i){
		Data[i] += m.Data[i];
	}
	return *this;
}

cpuMat& cpuMat::operator+=(float val) {
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	for(int i = 0; i < n; ++i){
		Data[i] += val;
	}
	return *this;
}

cpuMat& cpuMat::operator+=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			Data[i + n * ch] += v.get(ch);
		}
	}
	return *this;
}

cpuMat cpuMat::operator-(const cpuMat &m) const{

	if(NULL == Data  || NULL == m.Data || getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(m);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] - tmp.Data[i];
	}
	return tmp;
}

cpuMat cpuMat::operator-(float val) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(rows, cols, channels);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] - val;
	}
	return tmp;
}

cpuMat cpuMat::operator-(const vector3f& v) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	cpuMat tmp(rows, cols, channels);
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			tmp.Data[i + n * ch] = Data[i + n * ch] - v.get(ch);
		}
	}
	return tmp;
}

cpuMat& cpuMat::operator-=(const cpuMat &m){
	if(NULL == Data  || NULL == m.Data|| getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	for(int i = 0; i < n; ++i){
		Data[i] -= m.Data[i];
	}
	return *this;
}

cpuMat& cpuMat::operator-=(float val) {
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	for(int i = 0; i < n; ++i){
		Data[i] -= val;
	}
	return *this;
}

cpuMat& cpuMat::operator-=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			Data[i + n * ch] -= v.get(ch);
		}
	}
	return *this;
}

//cpuMat cpuMat::operator*(const cpuMat &m){}


cpuMat cpuMat::operator*(float val) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(rows, cols, channels);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] * val;
	}
	return tmp;
}

cpuMat cpuMat::operator*(const vector3f &v) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	cpuMat tmp(rows, cols, channels);
	for(int ch = 0; ch < 3; ++ch){
		for(int i = 0; i < channels; ++i){
			tmp.Data[i + n * ch] = Data[i + n * ch] * v.get(ch);
		}
	}
	return tmp;
}

cpuMat& cpuMat::operator*=(float val) {
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	for(int i = 0; i < n; ++i){
		Data[i] *= val;
	}
	return *this;
}

cpuMat& cpuMat::operator*=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			Data[i + n * ch] *= v.get(ch);
		}
	}
	return *this;
}

cpuMat cpuMat::mul(const cpuMat &m) const{
	if(NULL == Data  || NULL == m.Data || getLength()!= m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(m);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] * tmp.Data[i];
	}
	return tmp;
}

cpuMat cpuMat::mul(float val) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cpuMat tmp(rows, cols, channels);
	for(int i = 0; i < n; ++i){
		tmp.Data[i] = Data[i] * val;
	}
	return tmp;
}

cpuMat cpuMat::mul(const vector3f &v) const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = rows * cols;
	cpuMat tmp(rows, cols, channels);
	for(int ch = 0; ch < channels; ++ch){
		for(int i = 0; i < n; ++i){
			tmp.Data[i + n * ch] = Data[i + n * ch] * v.get(ch);
		}
	}
	return tmp;
}


cpuMat cpuMat::t() const{
	if(NULL == Data ){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	cpuMat tmp(cols, rows, channels);
	for(int i = 0; i < tmp.rows; ++i){
		for(int j = 0; j < tmp.cols; ++j){
			tmp.set(i, j, get(j, i));
		}
	}
	return tmp;
}

// memory
void cpuMat::mallocMat(){
	if(NULL == Data){
		// malloc host data
		Data = (float*)MemoryMonitor::instance()->cpuMalloc(cols * rows * channels * sizeof(float));
		if(NULL == Data) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(Data, 0, cols * rows * channels * sizeof(float));
	}
}

void cpuMat::print(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == Data ){
		std::cout<<"invalid cpuMatrix..."<<std::endl;
		exit(0);
	}
	int counter = 0;
	std::cout<<"cpuMatrix with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<"columns."<<std::endl;
	for(int i = 0; i < channels; ++i){
		std::cout<<"Channel "<<i<<" : "<<std::endl;
		for(int j = 0; j < rows; ++j){
			for(int k = 0; k < cols; ++k){
				std::cout<<Data[counter]<<" ";
				++ counter;
			}
			std::cout<<std::endl;
		}
	}
}

void cpuMat::printDim(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == Data ){
		std::cout<<"invalid cpuMatrix..."<<std::endl;
		exit(0);
	}
	cout<<"Matrix Dimension = ["<<rows<<", "<<cols<<", "<<channels<<"]"<<endl;
}
