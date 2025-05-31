#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mnist_data {
    double data[28][28];
    unsigned int label;
};

static inline void swap_endianness(unsigned int* v) {
    *v = (*v >> 24) | ((*v << 8) & 0x00FF0000) | ((*v >> 8) & 0x0000FF00) | (*v << 24);
}

int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data, unsigned int *n) {
    unsigned char tmp[4];
    unsigned char read_data[28*28];
    FILE *ifp = fopen(image_filename, "rb");
    FILE *lfp = fopen(label_filename, "rb");

    if (!ifp || !lfp) {
        printf("Cannot open input files\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
        return -1;
    }

    // Đọc và kiểm tra header của tệp ảnh
    size_t ret = fread(tmp, 1, 4, ifp);
    if (ret != 4) { printf("Failed to read image magic number\n"); fclose(ifp); fclose(lfp); return -1; }

    ret = fread(tmp, 1, 4, ifp);
    if (ret != 4) { printf("Failed to read number of images\n"); fclose(ifp); fclose(lfp); return -1; }
    unsigned int num_images = *(unsigned int*)tmp;
    swap_endianness(&num_images);

    ret = fread(tmp, 1, 4, ifp);
    if (ret != 4) { printf("Failed to read number of rows\n"); fclose(ifp); fclose(lfp); return -1; }
    unsigned int num_rows = *(unsigned int*)tmp;
    swap_endianness(&num_rows);

    ret = fread(tmp, 1, 4, ifp);
    if (ret != 4) { printf("Failed to read number of cols\n"); fclose(ifp); fclose(lfp); return -1; }
    unsigned int num_cols = *(unsigned int*)tmp;
    swap_endianness(&num_cols);

    // Đọc và kiểm tra header của tệp nhãn
    ret = fread(tmp, 1, 4, lfp);
    if (ret != 4) { printf("Failed to read label magic number\n"); fclose(ifp); fclose(lfp); return -1; }

    ret = fread(tmp, 1, 4, lfp);
    if (ret != 4) { printf("Failed to read number of labels\n"); fclose(ifp); fclose(lfp); return -1; }
    unsigned int num_labels = *(unsigned int*)tmp;
    swap_endianness(&num_labels);

    if (num_images != num_labels || num_rows != 28 || num_cols != 28) {
        printf("Invalid MNIST data\n");
        fclose(ifp);
        fclose(lfp);
        return -1;
    }

    *n = num_images;
    *data = (mnist_data*)malloc(num_images * sizeof(mnist_data));
    if (!*data) {
        printf("Memory allocation failed\n");
        fclose(ifp);
        fclose(lfp);
        return -1;
    }

    for (unsigned int i = 0; i < num_images; i++) {
        ret = fread(read_data, 1, 28*28, ifp);
        if (ret != 28*28) { printf("Failed to read image %u\n", i); free(*data); fclose(ifp); fclose(lfp); return -1; }
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                (*data)[i].data[r][c] = read_data[r*28 + c] / 255.0;
            }
        }

        ret = fread(tmp, 1, 1, lfp);
        if (ret != 1) { printf("Failed to read label %u\n", i); free(*data); fclose(ifp); fclose(lfp); return -1; }
        (*data)[i].label = tmp[0];
    }

    fclose(ifp);
    fclose(lfp);
    return 0;
}