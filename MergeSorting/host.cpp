#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "CL/cl.hpp"

using namespace std;

#define SIZE 1024

cl::Device* getDevice() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> allDevices;

    for (auto& p : platforms) {
        vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }

        devices.clear();

        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }
    }

    int choice = -1;
    do {
        cout << "Select a device to use from the following list.." << endl << endl;
        for (int i = 0; i < allDevices.size(); i++) {
            auto& d = allDevices[i];
            cout << "Device #: " << i + 1 << endl;
            cout << "Device Name: " << d.getInfo<CL_DEVICE_NAME>() << endl;
            cout << "Device Type: " << (d.getInfo<CL_DEVICE_TYPE>() == 2 ? "CPU" : "GPU") << endl << endl;
        }
        cout << "Enter device #: ";
        cin >> choice;
        cin.ignore();
        cout << endl;
    } while (choice < 1 || choice > allDevices.size());

    return new cl::Device(allDevices[choice - 1]);
}

void initializeArray(int* _vector) {
    for (int i = 0; i < SIZE; i++) {
        _vector[i] = rand() % SIZE;
    }
}

void compAndSwap(int a[], int i, int j, int dir) {
    if (dir == (a[i] > a[j]))
        swap(a[i], a[j]);
}

void bitonicMerge(int* a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            compAndSwap(a, i, i + k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

void bitonicSort(int* a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;

        // sort in ascending order since dir here is 1
        bitonicSort(a, low, k, 1);

        // sort in descending order since dir here is 0
        bitonicSort(a, low + k, k, 0);

        // Will merge whole sequence in ascending order since dir=1.
        bitonicMerge(a, low, cnt, dir);
    }
}

void sort(int* a, int N) {
    bitonicSort(a, 0, N, 1);
}


int main() {
    srand(time(0));
    int* data = new int[SIZE];
    cout << "Initializing Array" << endl;
    initializeArray(data);
    cout << "Array Initialized" << endl << endl;

    cl::Device* device = getDevice();
    cout << "Selected Device: " << device->getInfo<CL_DEVICE_NAME>() << endl;

    int* serialData = new int[SIZE];
    copy(data, data + SIZE, serialData);

    auto start = chrono::high_resolution_clock::now();
    sort(serialData, SIZE);
    auto end = chrono::high_resolution_clock::now();
    double serialDuration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    ifstream file("kernel.cl");
    string src(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(*device);
    cl::Program program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device)
            << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << std::endl;
        exit(1);
    }

    cl::Buffer dataBuffer(context, CL_MEM_READ_WRITE, SIZE * sizeof(int));

    cl::Kernel kernel(program, "Sorting");
    kernel.setArg(0, dataBuffer);
    kernel.setArg(1, SIZE);

	cl::CommandQueue queue(context, *device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, SIZE * sizeof(int), data);

    double parallelDuration = 0;
    for (unsigned int k = 2; k <= SIZE; k <<= 1) {
        for (unsigned int j = k / 2; j > 0; j >>= 1) {
            kernel.setArg(2, j);
            kernel.setArg(3, k);

            start = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(SIZE), cl::NullRange, NULL, NULL);
            queue.finish();
            end = chrono::high_resolution_clock::now();
            parallelDuration += chrono::duration_cast<chrono::microseconds>(end - start).count();
        }
    }

    {
        parallelDuration /= 100;
    }

    queue.enqueueReadBuffer(dataBuffer, CL_TRUE, 0, SIZE * sizeof(int), data);

    for (int i = 0; i < SIZE; i++) {
        if (data[i] != serialData[i]) {
			cerr << "Error at index " << i << " Expected: " << serialData[i] << " Actual: " << data[i] << endl;
	    	return 1;
		}
	}

    cout << "Serial Duration: " << serialDuration << " microseconds" << endl;
    cout << "Parallel Duration: " << parallelDuration << " microseconds" << endl;

    delete[] data;
    delete[] serialData;

    return 0;
}

// For Graph Generation
/*
vector<cl::Device> getAllDevices() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    vector<cl::Device> allDevices;

    for (auto& p : platforms) {
        vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }

        devices.clear();

        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        for (auto& d : devices) {
            allDevices.push_back(d);
        }
    }

    return allDevices;
}

double runParallel(cl::Device* device, int* _data, int* localData) {
    int *data = new int[SIZE];
    copy(_data, _data + SIZE, data);
    ifstream file("kernel.cl");
    string src(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(*device);
    cl::Program program(context, sources);

    auto err = program.build();
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*device)
            << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*device) << std::endl;
        exit(1);
    }

    cl::Buffer dataBuffer(context, CL_MEM_READ_WRITE, SIZE * sizeof(int));

    cl::Kernel kernel(program, "Sorting");
    kernel.setArg(0, dataBuffer);
    kernel.setArg(1, SIZE);

    cl::CommandQueue queue(context, *device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, SIZE * sizeof(int), data);

    chrono::high_resolution_clock::time_point start, end;
    double parallelDuration = 0;
    for (unsigned int k = 2; k <= SIZE; k <<= 1) {
        for (unsigned int j = k / 2; j > 0; j >>= 1) {
            kernel.setArg(2, j);
            kernel.setArg(3, k);

            start = chrono::high_resolution_clock::now();
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(SIZE), cl::NullRange, NULL, NULL);
            queue.finish();
            end = chrono::high_resolution_clock::now();
            parallelDuration += chrono::duration_cast<chrono::microseconds>(end - start).count();
        }
    }

    queue.enqueueReadBuffer(dataBuffer, CL_TRUE, 0, SIZE * sizeof(int), data);

    for (int i = 0; i < SIZE; i++) {
        if (data[i] != localData[i]) {
            cerr << "Error at index " << i << " Expected: " << localData[i] << " Actual: " << data[i] << endl;
            return 1;
        }
    }

    delete[] data;
    return parallelDuration;
}

int main() {
    vector<cl::Device> allDevices = getAllDevices();
    vector<double> durations;

    int* data = new int[SIZE];
    cout << "Initializing Array" << endl;
    initializeArray(data);
    cout << "Array Initialized" << endl << endl;

    int* localData = new int[SIZE];
    copy(data, data + SIZE, localData);

    auto start = chrono::high_resolution_clock::now();
    sort(localData, SIZE);
    auto end = chrono::high_resolution_clock::now();
    double serialDuration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout << "Serial Duration: " << serialDuration << " microseconds" << endl;
    durations.push_back(serialDuration);

    for (auto& d : allDevices) {
        double parallelDuration = runParallel(&d, data, localData);
        cout << d.getInfo<CL_DEVICE_NAME>() << " Duration: " << parallelDuration << " microseconds" << endl;
		durations.push_back(parallelDuration);
	}

	//ofstream file("results.csv", ios::app);
	//file << "SIZE,Serial,Parallel CPU,Parallel Intel GPU,Parallel Nvidia GPU" << endl;
    //file << SIZE << ",";
    //for (int i = 0; i < durations.size(); i++) {
    //    if (i == durations.size() - 1)
	//		file << durations[i] << endl;
	//	else
	//		file << durations[i] << ",";
    //}

    delete[] data;
    delete[] localData;
}
*/