#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <string>

typedef unsigned char uchar;

class MNISTReader {
   public:
    // Constructor: loads the MNIST images from the given file path
    MNISTReader(const std::string& full_path);

    // Destructor: frees the memory used for the images
    ~MNISTReader();

    // Function to get the number of images
    int get_number_of_images() const;

    // Function to get the image size (number of pixels in each image)
    int get_image_size() const;

    // Function to get a pointer to a specific image
    uchar* get_image(int index) const;

   private:
    int number_of_images;
    int image_size;
    uchar** dataset;

    // Helper function to reverse the byte order of integers
    static int reverseInt(int i);

    // Function to load the MNIST images from the given file
    void load_images(const std::string& full_path);
};

#endif