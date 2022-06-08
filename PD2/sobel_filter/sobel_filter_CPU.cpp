/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2022 Riga Technical University
 */
#include "sobel_filter_CPU.h"

#include <ctime>
#include "sobel_filter_common.h"

#include <iostream>

void sobelFilter8BPPPalette(const BYTE* inputImage, const DWORD* paletteColors, BYTE* outputImage, int width, int height) {
	//DWORD window[WINDOW_SIZE];

	for (int j = 0; j < height - 2; j++) {
		for (int i = 0; i < width - 2; i++) {

			//computing the sobel gradient in x and y direction
			int g_x = -1 * inputImage[j, i] + -2 * inputImage[j + 1, i]
				+ 1 * inputImage[j + 2, i] + 0 * inputImage[j, i + 1]
				+ 0 * inputImage[j + 1, i + 1] + 0 * inputImage[j + 2, i + 1]
				+ 1 * inputImage[j, i + 2] + 2 * inputImage[j + 1, i + 2]
				+ 1 * inputImage[j + 2, i + 2];

			int g_y = 1 * inputImage[j, i] + 2 * inputImage[j + 1, i]
				+ 1 * inputImage[j + 2, i] + 0 * inputImage[j, i + 1]
				+ 0 * inputImage[j + 1, i + 1] + 0 * inputImage[j + 2, i + 1]
				+ -1 * inputImage[j, i + 2] + -2 * inputImage[j + 1, i + 2]
				+ -1 * inputImage[j + 2, i + 2];

			int sum = abs(g_x) + abs(g_y);
			//if (sum > 255) {
			//	sum = 255;
			//}
			outputImage[i * width + j] = sum;
		}
	}

}

void sobelFilterRGBA(const DWORD* inputImage, DWORD* outputImage, int width, int height) {
	//DWORD window[WINDOW_SIZE];

	for (int j = 0; j < width - 2; j++) {
		for (int i = 0; i < height - 2; i++) {

			//computing the sobel gradient in x and y direction
			float g_x = -1 * inputImage[j, i] + -2 * inputImage[j + 1, i]
				+ 1 * inputImage[j + 2, i] + 0 * inputImage[j, i + 1]
				+ 0 * inputImage[j + 1, i + 1] + 0 * inputImage[j + 2, i + 1]
				+ 1 * inputImage[j, i + 2] + 2 * inputImage[j + 1, i + 2]
				+ 1 * inputImage[j + 2, i + 2];

			float g_y = 1 * inputImage[j, i] + 2 * inputImage[j + 1, i]
				+ 1 * inputImage[j + 2, i] + 0 * inputImage[j, i + 1]
				+ 0 * inputImage[j + 1, i + 1] + 0 * inputImage[j + 2, i + 1]
				+ -1 * inputImage[j, i + 2] + -2 * inputImage[j + 1, i + 2]
				+ -1 * inputImage[j + 2, i + 2];

			float sum = sqrt(g_x*g_x + g_y*g_y);
			//if (sum > 255) {
			//	sum = 255;
			//}
			outputImage[i * width + j] = sum;
		}
	}

}

bool sobelFilterGrayscaleCPU(const BitmapImage& inputImage, const BitmapImage& outputImage) {
	using namespace std;

	clock_t begin = clock();
	
	if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
		sobelFilter8BPPPalette(inputImage.getRawData(), (DWORD*)inputImage.getPaletteColors(), outputImage.getRawData(), inputImage.getWidth(), inputImage.getHeight());
	} else {
		sobelFilterRGBA((DWORD*)inputImage.getRawData(), (DWORD*)outputImage.getRawData(), inputImage.getWidth(), inputImage.getHeight());
	}
		
	clock_t end = clock();
	float elapsedTime = float(end - begin) / CLOCKS_PER_SEC;
	cout << "---------------------" << endl;
	cout << "Elapsed image processing time on CPU: " << elapsedTime << " seconds" << endl;
	cout << "---------------------" << endl;
	
	return true;
}
