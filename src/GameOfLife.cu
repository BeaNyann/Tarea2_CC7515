// includes, system
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <random>

// includes CUDA
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

bool gridExists = false;

size_t gridHeight;
size_t gridWidth;
size_t gridSize;
// Host
char *h_grid;
char *h_auxGrid;
// Device
char *d_grid;
char *d_auxGrid;

#pragma region UTILITY
// Prints an array to standard output.
void printArray(char *arr, size_t size) {
	std::cout << "{ ";
	for (size_t i = 0; i < size; i++) {
		std::cout << +arr[i];
		if (i != size - 1) {
			std::cout << ", ";
		}
	}
	std::cout << " }";
}

// Checks if two arrays are equal.
bool arrayEquals(char *a, char *b, size_t size) {
	for (size_t i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

// Copies an array from src to dst.
void arrayCopy(char *src, char *dst, size_t size) {
	for (size_t i = 0; i < size; i++) {
		dst[i] = src[i];
	}
}
#pragma endregion

#pragma region GRID
/// Deletes the game grid.
void deleteGrid() {
	if (gridExists) {
		delete[] h_grid;
		delete[] h_auxGrid;
		gridSize = 0;
		gridHeight = 0;
		gridWidth = 0;
		gridExists = false;
	}
}

/// Fills an array with random values in the range [0, 1].
void initArray(char * data, int size) {
	std::random_device randomDevice;
	std::mt19937 generator(randomDevice());
	std::uniform_int_distribution<int> distr(0, 1);
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		data[i] = distr(generator);
	}
}

/// Creates a new grid with dimensions height * width.
void initGrid(size_t height, size_t width, bool verbose) {
	if (verbose) {
		std::cout << "Generating a new grid of " << width << " x " << height
			<< " cells." << std::endl;
	}
	deleteGrid();
	gridHeight = height;
	gridWidth = width;
	gridSize = height * width;
	h_grid = new char[gridSize];
	h_auxGrid = new char[gridSize];
	initArray(h_grid, gridSize);
	gridExists = true;
}

// Creates a new game grid from an array.
void arrayToGrid(char *src, size_t width, size_t height) {
	deleteGrid();
	gridHeight = height;
	gridWidth = width;
	gridSize = height * width;
	h_grid = new char[gridSize];
	h_auxGrid = new char[gridSize];
	arrayCopy(src, h_grid, gridSize);
	gridExists = true;
}

/// Prints a given grid to console.
void printGrid(char *grid, size_t width, size_t height) {
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++) {
			if (grid[row * width + col] == 1)
				std::cout << " O ";
			else
				std::cout << "   ";
		}
		std::cout << std::endl;
	}
}

/// Prints the game of life grid to console.
void printGrid() {
	printGrid(h_grid, gridWidth, gridHeight);
}
#pragma endregion

#pragma region GOL_CPU
void iterateCPU(size_t iterations) {
	for (size_t it = 0; it < iterations; it++) {
		for (size_t cellId = 0; cellId < gridSize; cellId++) {
			unsigned int x = cellId % gridWidth;
			unsigned int y = cellId - x;
			unsigned int xLeft = (x + gridWidth - 1) % gridWidth;
			unsigned int xRight = (x + 1) % gridWidth;
			unsigned int yUp = (y + gridSize - gridWidth) % gridSize;
			unsigned int yDown = (y + gridWidth) % gridSize;

			unsigned int aliveCells = h_grid[xLeft + yUp] + h_grid[x + yUp]
				+ h_grid[xRight + yUp] + h_grid[xLeft + y] + h_grid[xRight + y]
				+ h_grid[xLeft + yDown] + h_grid[x + yDown] + h_grid[xRight + yDown];
			h_auxGrid[x + y] = aliveCells == 3
				|| (aliveCells == 2 && h_grid[x + y]) ? 1 : 0;
		}
		std::swap(h_grid, h_auxGrid);
	}
}
#pragma endregion

#pragma region GOL_GPU
// Kernel that computes an iterarion of the game of life.
__global__ void gameOfLifeKernel(char *golGrid,
	unsigned int width, unsigned int height, char *auxGolGrid) {

	unsigned int size = width * height;
	for (unsigned int cellId = blockIdx.x * blockDim.x + threadIdx.x;
		cellId < size;
		cellId += blockDim.x * gridDim.x) {
		unsigned int x = cellId % width;
		unsigned int y = cellId - x;
		unsigned int xLeft = (x + width - 1) % width;
		unsigned int xRight = (x + 1) % width;
		unsigned int yUp = (y + size - width) % size;
		unsigned int yDown = (y + width) % size;

		unsigned int aliveCells = golGrid[xLeft + yUp] + golGrid[x + yUp]
			+ golGrid[xRight + yUp] + golGrid[xLeft + y] + golGrid[xRight + y]
			+ golGrid[xLeft + yDown] + golGrid[x + yDown] + golGrid[xRight + yDown];
		auxGolGrid[x + y] = aliveCells == 3
			|| (aliveCells == 2 && golGrid[x + y]) ? 1 : 0;
	}
}

// Simulates the game of life in GPU.
void iterateGPU(size_t iterations, unsigned short numThreads) {
	unsigned int memSizeGrid = sizeof(char) * gridSize;
	// Allocate device memory
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_grid), memSizeGrid));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_auxGrid), memSizeGrid));
	// copy host memory to device
	checkCudaErrors(
		cudaMemcpy(d_grid, h_grid, memSizeGrid, cudaMemcpyHostToDevice));
	checkCudaErrors(
		cudaMemcpy(d_auxGrid, h_auxGrid, memSizeGrid, cudaMemcpyHostToDevice));
	size_t reqBlocks = gridWidth * gridHeight / numThreads;
	unsigned short blocksCount
		= (unsigned short)std::min((size_t)32768, reqBlocks);
	for (size_t i = 0; i < iterations; i++) {
		gameOfLifeKernel<<<blocksCount, numThreads>>>(d_grid, gridWidth,
			gridHeight, d_auxGrid);
		std::swap(d_grid, d_auxGrid);
	}
	// Copy result from device to host
	checkCudaErrors(
		cudaMemcpy(h_grid, d_grid, memSizeGrid, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_grid));
	checkCudaErrors(cudaFree(d_auxGrid));
}

// Simulates the game of life in GPU with a given block size.
void iterateGPU_BS(size_t iterations, unsigned short blockSize) {

	unsigned int memSizeGrid = sizeof(char) * gridSize;
	// Allocate device memory
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_grid), memSizeGrid));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_auxGrid), memSizeGrid));
	// copy host memory to device
	checkCudaErrors(
		cudaMemcpy(d_grid, h_grid, memSizeGrid, cudaMemcpyHostToDevice));
	checkCudaErrors(
		cudaMemcpy(d_auxGrid, h_auxGrid, memSizeGrid, cudaMemcpyHostToDevice));
	// Setup execution parameters
	size_t reqThreads = std::max((size_t)2, (gridWidth * gridHeight) / blockSize);
	unsigned short threads = 64;

	for (size_t i = 0; i < iterations; i++) {
		gameOfLifeKernel <<<blockSize, threads>>>(d_grid, gridWidth,
			gridHeight, d_auxGrid);
		std::swap(d_grid, d_auxGrid);
	}
	// Copy result from device to host
	checkCudaErrors(
		cudaMemcpy(h_grid, d_grid, memSizeGrid, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_grid));
	checkCudaErrors(cudaFree(d_auxGrid));
}

// Kernel that computes an iteration of the game of life using ifs to check for
// alive neighbours.
__global__ void gameOfLifeIfKernel(char *golGrid,
	unsigned int width, unsigned int height, char *auxGolGrid) {

	unsigned int size = width * height;
	for (unsigned int cellId = blockIdx.x * blockDim.x + threadIdx.x;
		cellId < size;
		cellId += blockDim.x * gridDim.x) {
		unsigned int x = cellId % width;
		unsigned int y = cellId - x;
		unsigned int xLeft = (x + width - 1) % width;
		unsigned int xRight = (x + 1) % width;
		unsigned int yUp = (y + size - width) % size;
		unsigned int yDown = (y + width) % size;
		unsigned int aliveCells = 0;

		if (golGrid[xLeft + yUp]) aliveCells++;
		if (golGrid[x + yUp]) aliveCells++;
		if (golGrid[xRight + yUp]) aliveCells++;
		if (golGrid[xLeft + y]) aliveCells++;
		if (golGrid[xRight + y]) aliveCells++;
		if (golGrid[xLeft + yDown]) aliveCells++;
		if (golGrid[x + yDown]) aliveCells++;
		if (golGrid[xRight + yDown]) aliveCells++;

		auxGolGrid[x + y] = aliveCells == 3
			|| (aliveCells == 2 && golGrid[x + y]) ? 1 : 0;
	}
}

// Simulates the game of life in GPU using ifs to check for alive neighbours.
void iterateGPU_IF(size_t iterations, unsigned short numThreads) {
	unsigned int memSizeGrid = sizeof(char) * gridSize;
	// Allocate device memory
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_grid), memSizeGrid));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_auxGrid), memSizeGrid));
	// copy host memory to device
	checkCudaErrors(
		cudaMemcpy(d_grid, h_grid, memSizeGrid, cudaMemcpyHostToDevice));
	checkCudaErrors(
		cudaMemcpy(d_auxGrid, h_auxGrid, memSizeGrid, cudaMemcpyHostToDevice));
	size_t reqBlocks = (gridWidth * gridHeight) / numThreads;
	unsigned short blocksCount
		= (unsigned short)std::min((size_t)32768, reqBlocks);
	for (size_t i = 0; i < iterations; i++) {
		gameOfLifeIfKernel<<<blocksCount, numThreads>>>(d_grid, gridWidth,
			gridHeight, d_auxGrid);
		std::swap(d_grid, d_auxGrid);
	}
	// Copy result from device to host
	checkCudaErrors(
		cudaMemcpy(h_grid, d_grid, memSizeGrid, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_grid));
	checkCudaErrors(cudaFree(d_auxGrid));
}
#pragma endregion

#pragma region Testing
// Tests if arrayEquals works correctly.
void testArrayEquals() {
	char *a = new char[5]{ 0, 1, 2, 3, 4 };
	char *b = new char[5]{ 0, 1, 2, 3, 4 };
	char *c = new char[5]{ 0, 1, 3, 3, 4 };
	bool success = true;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: arrayEquals" << std::endl << std::endl;

	std::cout << "a: ";
	printArray(a, 5);
	std::cout << std::endl;
	std::cout << "b: ";
	printArray(b, 5);
	std::cout << std::endl;
	std::cout << "c: ";
	printArray(c, 5);
	std::cout << std::endl << std::endl;

	if (!arrayEquals(a, b, 5)) {
		std::cout << "TEST STATUS: FAILED: Arrays a and b should be equals!"
			<< std::endl;
		success = false;
	}
	if (arrayEquals(a, c, 5)) {
		std::cout << "TEST STATUS: FAILED: Arrays a and c should not be equals!" << std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	delete a;
	delete b;
	delete c;
}

// Tests if arrayCopy works properly.
void testArrayCopy() {
	bool success = true;
	char *a = new char[5]{ 0, 1, 2, 3, 4 };
	char *b = new char[5];

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: arrayCopy" << std::endl << std::endl;

	arrayCopy(a, b, 5);

	std::cout << "a: ";
	printArray(a, 5);
	std::cout << std::endl;
	std::cout << "b: ";
	printArray(b, 5);
	std::cout << std::endl << std::endl;

	if (!arrayEquals(a, b, 5)) {
		std::cout << "TEST STATUS: FAILED: Arrays a and b should be equals!"
			<< std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	delete a;
	delete b;
}

// Tests game of life execution in CPU.
void testIterateCPU() {
	bool success = true;
	char *iter0Grid = new char[64]{	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0	};
	char *iter1Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter2Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: GOL result in CPU." << std::endl << std::endl;

	arrayToGrid(iter0Grid, 8, 8);
	iterateCPU(0);
	if (!arrayEquals(h_grid, iter0Grid, 64)) {
		std::cout << "Expected grid after 0 iterations:" << std::endl;
		printGrid(iter0Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 0 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateCPU(1);
	if (!arrayEquals(h_grid, iter1Grid, 64)) {
		std::cout << "Expected grid after 1 iteration:" << std::endl;
		printGrid(iter1Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 1 iteration:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateCPU(2);
	if (!arrayEquals(h_grid, iter2Grid, 64)) {
		std::cout << "Expected grid after 2 iterations:" << std::endl;
		printGrid(iter2Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 2 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	deleteGrid();
	delete iter0Grid;
	delete iter1Grid;
	delete iter2Grid;
}

// Tests game of life execution in GPU.
void testIterateGPU(unsigned short threads) {
	bool success = true;
	char *iter0Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter1Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter2Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: GOL result in GPU with " << threads << " threads."
		<< std::endl << std::endl;

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU(0, threads);
	if (!arrayEquals(h_grid, iter0Grid, 64)) {
		std::cout << "Expected grid after 0 iterations:" << std::endl;
		printGrid(iter0Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 0 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU(1, threads);
	if (!arrayEquals(h_grid, iter1Grid, 64)) {
		std::cout << "Expected grid after 1 iteration:" << std::endl;
		printGrid(iter1Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 1 iteration:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU(2, threads);
	if (!arrayEquals(h_grid, iter2Grid, 64)) {
		std::cout << "Expected grid after 2 iterations:" << std::endl;
		printGrid(iter2Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 2 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	deleteGrid();
	delete iter0Grid;
	delete iter1Grid;
	delete iter2Grid;
}

// Tests game of life using ifs execution in GPU.
void testIterateGPU_IF(unsigned short threads) {
	bool success = true;
	char *iter0Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter1Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter2Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: GOL result in GPU-If with " << threads << " threads."
		<< std::endl << std::endl;

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_IF(0, threads);
	if (!arrayEquals(h_grid, iter0Grid, 64)) {
		std::cout << "Expected grid after 0 iterations:" << std::endl;
		printGrid(iter0Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 0 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_IF(1, threads);
	if (!arrayEquals(h_grid, iter1Grid, 64)) {
		std::cout << "Expected grid after 1 iteration:" << std::endl;
		printGrid(iter1Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 1 iteration:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_IF(2, threads);
	if (!arrayEquals(h_grid, iter2Grid, 64)) {
		std::cout << "Expected grid after 2 iterations:" << std::endl;
		printGrid(iter2Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 2 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	deleteGrid();
	delete iter0Grid;
	delete iter1Grid;
	delete iter2Grid;
}

// Tests game of life execution in GPU.
void testIterateGPU_BS(unsigned short blockSize) {
	bool success = true;
	char *iter0Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter1Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 1, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };
	char *iter2Grid = new char[64]{ 0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 1, 0, 0, 0, 0,
																	0, 1, 0, 1, 0, 0, 0, 0,
																	0, 0, 1, 1, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0,
																	0, 0, 0, 0, 0, 0, 0, 0 };

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: GOL result in GPU with block size " << blockSize << "."
		<< std::endl << std::endl;

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_BS(0, blockSize);
	if (!arrayEquals(h_grid, iter0Grid, 64)) {
		std::cout << "Expected grid after 0 iterations:" << std::endl;
		printGrid(iter0Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 0 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_BS(1, blockSize);
	if (!arrayEquals(h_grid, iter1Grid, 64)) {
		std::cout << "Expected grid after 1 iteration:" << std::endl;
		printGrid(iter1Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 1 iteration:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}

	arrayToGrid(iter0Grid, 8, 8);
	iterateGPU_BS(2, blockSize);
	if (!arrayEquals(h_grid, iter2Grid, 64)) {
		std::cout << "Expected grid after 2 iterations:" << std::endl;
		printGrid(iter2Grid, 8, 8);
		std::cout << std::endl;
		std::cout << "Resulting grid after 2 iterations:" << std::endl;
		printGrid();
		std::cout << std::endl;
		std::cout << "TEST STATUS : FAILED: Resulting grid is not equal to expected!"
			<< std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	deleteGrid();
	delete iter0Grid;
	delete iter1Grid;
	delete iter2Grid;
}

// Creates a test grid of size width x height.
void buildTestGrid(size_t width, size_t height) {
	deleteGrid();
	gridHeight = height;
	gridWidth = width;
	gridSize = height * width;
	h_grid = new char[gridSize];
	h_auxGrid = new char[gridSize];
	for (size_t i = 0; i < gridSize; i++) {
		h_grid[i] = (i % 5) <= 1 ? 1 : 0;
	}
	gridExists = true;
}

// Checks if the results obtained from running the game of life in CPU matches
// the ones obtained in GPU on a width x height grid running it iterations.
void testResults(size_t width, size_t height, size_t it) {
	bool success = true;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "TESTING: GOL results in CPU and GPU on a " << width << " x "
		<< height << " grid and " << it << " iterations." << std::endl << std::endl;

	buildTestGrid(width, height);
	// CPU execution
	char *cpuResult = new char[gridSize];
	iterateCPU(it);
	arrayCopy(h_grid, cpuResult, gridSize);
	// GPU execution
	buildTestGrid(width, height);
	char *gpuResult = new char[gridSize];
	iterateGPU(it, 32);
	arrayCopy(h_grid, gpuResult, gridSize);

	if (!arrayEquals(cpuResult, gpuResult, gridSize)) {
		std::cout << "CPU Result:" << std::endl;
		printGrid(cpuResult, width, height);
		std::cout << std::endl;
		std::cout << "GPU Result:" << std::endl;
		printGrid(gpuResult, width, height);
		std::cout << std::endl;
		std::cout << "TEST STATUS: FAILED: Result from iterating in cpu differs "
			<< "from result processed in gpu." << std::endl;
		success = false;
	}
	delete gpuResult;
	// GPU with ifs execution
	buildTestGrid(width, height);
	gpuResult = new char[gridSize];
	iterateGPU_IF(it, 32);
	arrayCopy(h_grid, gpuResult, gridSize);

	if (!arrayEquals(cpuResult, gpuResult, gridSize)) {
		std::cout << "CPU Result:" << std::endl;
		printGrid(cpuResult, width, height);
		std::cout << std::endl;
		std::cout << "GPU-If Result:" << std::endl;
		printGrid(gpuResult, width, height);
		std::cout << std::endl;
		std::cout << "TEST STATUS: FAILED: Result from iterating in cpu differs "
			<< "from result processed in gpu." << std::endl;
		success = false;
	}
	delete gpuResult;
	// GPU-BS execution
	buildTestGrid(width, height);
	gpuResult = new char[gridSize];
	iterateGPU_BS(it, 25);
	arrayCopy(h_grid, gpuResult, gridSize);

	if (!arrayEquals(cpuResult, gpuResult, gridSize)) {
		std::cout << "CPU Result:" << std::endl;
		printGrid(cpuResult, width, height);
		std::cout << std::endl;
		std::cout << "GPU-BS Result:" << std::endl;
		printGrid(gpuResult, width, height);
		std::cout << std::endl;
		std::cout << "TEST STATUS: FAILED: Result from iterating in cpu differs "
			<< "from result processed in gpu." << std::endl;
		success = false;
	}
	if (success) {
		std::cout << "TEST STATUS: PASSED." << std::endl;
	}
	deleteGrid();
	delete gpuResult;
	delete cpuResult;
}

// Runs a set of tests.
void runTests() {
	testArrayEquals();
	testArrayCopy();
	// CPU
	testIterateCPU();
	// GPU
	testIterateGPU(1);
	testIterateGPU(16);
	testIterateGPU(32);
	// GPU-IF
	testIterateGPU_IF(1);
	testIterateGPU_IF(16);
	testIterateGPU_IF(32);
	// GPU-BS
	testIterateGPU_BS(25);
	testIterateGPU_BS(32);
	testIterateGPU_BS(81);
	// GPU-CPU
	testResults(32, 32, 1);
	testResults(16, 16, 4);
	testResults(8, 8, 16);
	testResults(128, 128, 32);
}
#pragma endregion

#pragma region BENCHMARKS
// Benchmarks performance of the game of life in CPU.
void benchmarkCPU(int startSize, int endSize, bool verbose) {
	std::string sep(100, '-');
	std::ofstream csvFile;
	csvFile.open("benchmarkCPU.csv");
	csvFile << "grid_size,cells/sec" << std::endl;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "BENCHMARKING: GOL CPU."
		<< std::endl << std::endl;

	size_t dim = startSize;
	while (dim <= endSize) {
		std::cout << sep << std::endl;
		std::cout << "Computing 16 iterations for a " << dim << " x " << dim
			<< " grid (" << (dim * dim) << " cells)." << std::endl;
		double cellsPerSec = 0.0;
		double meanTime = 0.0;
		for (size_t i = 0; i < 5; i++) {
			initGrid(dim, dim, verbose);
			auto startTime = std::chrono::high_resolution_clock::now();
			iterateCPU(16);
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> delta = endTime - startTime;
			cellsPerSec += gridSize / delta.count();
			meanTime += delta.count();
		}
		cellsPerSec /= 5;
		csvFile << gridSize << "," << cellsPerSec << std::endl;
		std::cout << "Mean elapsed time: " << meanTime << " s" << std::endl;
		std::cout << "Cells/sec: " << cellsPerSec << std::endl;
		dim *= 2;
	}
	csvFile.close();
	deleteGrid();
}

// Benchmark performance of the game of life in GPU.
void benchmarkGPU(int startSize, int endSize, unsigned short threads,
	bool verbose) {
	std::string sep(100, '-');
	std::ofstream csvFile;
	std::ostringstream filename;
	filename << "benchmarkGPU_x" << threads << "th.csv";
	csvFile.open(filename.str());
	csvFile << "grid_size,cells/sec" << std::endl;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "BENCHMARKING: GOL GPU on " << threads << " threads."
		<< std::endl << std::endl;

	size_t dim = startSize;
	while (dim <= endSize) {
		std::cout << sep << std::endl;
		std::cout << "Computing 16 iterations for a " << dim << " x " << dim
			<< " grid (" << (dim * dim) << " cells)." << std::endl;
		std::cout << "threads: " << threads << std::endl;
		size_t reqBlocks = (dim * dim) / threads;
		unsigned short blockSize
			= (unsigned short)std::min((size_t)32768, reqBlocks);
		std::cout << "block_size: " << blockSize << std::endl;
		double cellsPerSec = 0.0;
		double meanTime = 0.0;

		for (size_t i = 0; i < 5; i++) {
			initGrid(dim, dim, verbose);
			auto startTime = std::chrono::high_resolution_clock::now();
			iterateGPU(16, threads);
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> delta = endTime - startTime;
			cellsPerSec += gridSize / delta.count();
			meanTime += delta.count();
		}
		cellsPerSec /= 5;
		csvFile << gridSize << "," << cellsPerSec << std::endl;
		std::cout << "Mean elapsed time: " << meanTime << " s" << std::endl;
		std::cout << "Cells/sec: " << cellsPerSec << std::endl;
		dim *= 2;
	}
	csvFile.close();
	deleteGrid();
}

// Benchmark performance of the game of life with ifs in GPU.
void benchmarkGPU_If(int startSize, int endSize, unsigned short threads,
	bool verbose) {
	std::string sep(100, '-');
	std::ofstream csvFile;
	std::ostringstream filename;
	filename << "benchmarkGPU-If_x" << threads << "th.csv";
	csvFile.open(filename.str());
	csvFile << "grid_size,cells/sec" << std::endl;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "BENCHMARKING: GOL GPU-If on " << threads << " threads."
		<< std::endl << std::endl;

	size_t dim = startSize;
	while (dim <= endSize) {
		std::cout << sep << std::endl;
		std::cout << "Computing 16 iterations for a " << dim << " x " << dim
			<< " grid (" << (dim * dim) << " cells)." << std::endl;
		std::cout << "threads: " << threads << std::endl;
		size_t reqBlocks = (dim * dim) / threads;
		unsigned short blockSize
			= (unsigned short)std::min((size_t)32768, reqBlocks);
		std::cout << "block_size: " << blockSize << std::endl;
		double cellsPerSec = 0.0;
		double meanTime = 0.0;
		for (size_t i = 0; i < 5; i++) {
			initGrid(dim, dim, verbose);
			auto startTime = std::chrono::high_resolution_clock::now();
			iterateGPU_IF(16, threads);
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> delta = endTime - startTime;
			cellsPerSec += gridSize / delta.count();
			meanTime += delta.count();
		}
		cellsPerSec /= 5;
		csvFile << gridSize << "," << cellsPerSec << std::endl;
		std::cout << "Mean elapsed time: " << meanTime << " s" << std::endl;
		std::cout << "Cells/sec: " << cellsPerSec << std::endl;
		dim *= 2;
	}
	csvFile.close();
	deleteGrid();
}

// Benchmark performance of the game of life with fixed block size in GPU.
void benchmarkGPU_BS(int startSize, int endSize, unsigned short blockSize,
	bool verbose) {

	std::string sep(100, '-');
	std::ofstream csvFile;
	std::ostringstream filename;
	filename << "benchmarkGPU-BS_x" << blockSize << "bs.csv";
	csvFile.open(filename.str());
	csvFile << "grid_size,cells/sec" << std::endl;

	std::cout << std::string(100, '=') << std::endl;
	std::cout << "BENCHMARKING: GOL GPU-BS with block size of " << blockSize << "."
		<< std::endl << std::endl;

	size_t dim = startSize;
	while (dim <= endSize) {
		std::cout << sep << std::endl;
		std::cout << "Computing 16 iterations for a " << dim << " x " << dim
			<< " grid (" << (dim * dim) << " cells)." << std::endl;
		size_t reqThreads = std::max((size_t)2, (dim * dim) / blockSize);
		unsigned short threads = (unsigned short)std::min((size_t)32768, reqThreads);
		std::cout << "threads: " << threads << std::endl;
		std::cout << "block_size: " << blockSize << std::endl;
		double cellsPerSec = 0.0;
		double meanTime = 0.0;
		for (size_t i = 0; i < 5; i++) {
			initGrid(dim, dim, verbose);
			auto startTime = std::chrono::high_resolution_clock::now();
			iterateGPU_BS(16, blockSize);
			auto endTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> delta = endTime - startTime;
			cellsPerSec += gridSize / delta.count();
			meanTime += delta.count();
		}
		cellsPerSec /= 5;
		csvFile << gridSize << "," << cellsPerSec << std::endl;
		std::cout << "Mean elapsed time: " << meanTime << " s" << std::endl;
		std::cout << "Cells/sec: " << cellsPerSec << std::endl;
		dim *= 2;
	}
	csvFile.close();
	deleteGrid();
}

// Runs benchmarks for the game of life with different implementations and
// configurations.
void runBenchmarks(int startSize, int endSize) {
	auto startTime = std::chrono::high_resolution_clock::now();
	// Performance de CPU
	//benchmarkCPU(startSize, endSize, false);
	//// Performance GPU-If
	//benchmarkGPU_If(startSize, endSize, 128, false);
	// Performance GPU tpb 32x
	benchmarkGPU(startSize, endSize, 1024, true);
	benchmarkGPU(startSize, endSize, 512, true);
	benchmarkGPU(startSize, endSize, 256, true);
	benchmarkGPU(startSize, endSize, 128, true);
	benchmarkGPU(startSize, endSize, 64, true);
	benchmarkGPU(startSize, endSize, 32, true);
	// Performance GPU tpb ~32x
	benchmarkGPU(startSize, endSize, 625, true);
	benchmarkGPU(startSize, endSize, 81, true);

	auto endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> delta = endTime - startTime;
	std::cout << std::endl;
	std::cout << "Total elapsed time: " << delta.count() << " s" << std::endl;
}
#pragma endregion

void runGame() {
	initGrid(25, 25, false);
	while (true) {
		if (system("CLS")) {
			system("clear");
		}
		std::cout << "[CUDA Game Of Life]" << std::endl;
		iterateGPU(1, 32);
		printGrid();
		_sleep(800);
	}
}

int main(int argc, char **argv) {
	std::cout << "[CUDA Game Of Life] - Starting..." << std::endl;
	//runTests();
	//runBenchmarks(std::pow(2, 5), std::pow(2, 15)); // Uncomment to measure performance
	_sleep(5000);
	runGame();
	return 0;
}