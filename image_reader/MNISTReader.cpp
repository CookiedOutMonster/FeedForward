#include "MNISTReader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

// constructor
MNISTReader::MNISTReader(const std::string& full_path) {
    load_images(full_path);
}

// destructor
MNISTReader::~MNISTReader() {
    for (int i = 0; i < number_of_images; i++) {
        delete[] dataset[i];
    }
    delete[] dataset;
}

int MNISTReader::get_number_of_images() const {
    return number_of_images;
}

int MNISTReader::get_image_size() const {
    return image_size;
}

uchar* MNISTReader::get_image(int index) const {
    if (index < 0 || index >= number_of_images) {
        throw std::out_of_range("Index out of range");
    }
    return dataset[index];
}

int MNISTReader::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void MNISTReader::load_images(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }

    int magic_number = 0, n_rows = 0, n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file!");
    }

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    image_size = n_rows * n_cols;

    // Allocate memory for the images
    dataset = new uchar*[number_of_images];
    for (int i = 0; i < number_of_images; i++) {
        dataset[i] = new uchar[image_size];
        file.read((char*)dataset[i], image_size);
    }
}