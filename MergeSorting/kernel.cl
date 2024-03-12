__kernel void Sorting(__global float* data, const unsigned int n, const unsigned int j, const unsigned int k) {
    unsigned int i = get_global_id(0);
    unsigned int ixj = i ^ j;
    if ((ixj) > i) {
        if ((i & k) == 0 && data[i] > data[ixj]) {
            float tmp = data[i];
            data[i] = data[ixj];
            data[ixj] = tmp;
        }
        if ((i & k) != 0 && data[i] < data[ixj]) {
            float tmp = data[i];
            data[i] = data[ixj];
            data[ixj] = tmp;
        }
    }
}