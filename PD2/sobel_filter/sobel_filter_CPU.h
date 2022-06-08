/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2022 Riga Technical University
 */
#pragma once

#include "bitmap_image.h"

bool sobelFilterGrayscaleCPU(const BitmapImage& inputImage, const BitmapImage& outputImage);
