#include <iostream>
#include <random>
#include <fstream>
#include <exception>



#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <Windows.h>

bool gridExists = false;

size_t gridHeight;
size_t gridWidth;
size_t gridSize;
// Host
char* h_grid;
char* h_auxGrid;
// Device
char* d_grid;
char* d_auxGrid;

// OpenCL things
int platform_id = 0, device_id = 0;
std::vector<cl::Platform> platforms;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Context context;
cl::Program program;
cl::Kernel gameOfLife_kernel;

int initializeOpenCL() {
	cl::Platform::get(&platforms);

	// Select the platform.
	platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

	// Create a context
	context = cl::Context(devices);

	// Create a command queue
	queue = cl::CommandQueue(context, devices[device_id]);

	std::ifstream sourceFile("kernel.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source = { sourceCode };

	program = cl::Program(context, source);


	try {
		// Build the program for the devices
		program.build(devices);
	}
	catch (cl::Error& e) {
		std::cout << "El proceso de inicialización falló con codigo:" << e.err() << std::endl;
		std::cout << e.what() << std::endl;
		size_t log_size;
		clGetProgramBuildInfo(program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		return 1;
	}
	// Make kernel
	gameOfLife_kernel = cl::Kernel(program, "gameOfLifeKernel");
	return 0;
}

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

void initArray(char* data, size_t size) {
	std::random_device randomDevice;
	std::mt19937 generator(randomDevice());
	std::uniform_int_distribution<int> distr(0, 1);
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++) {
		data[i] = distr(generator);
	}
}

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


void iterateGPU(size_t iterations, unsigned short numThreads) {
	size_t	 memSizeGrid = sizeof(char) * gridSize;

	cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_WRITE, memSizeGrid);
	cl::Buffer d_auxGrid = cl::Buffer(context, CL_MEM_READ_WRITE, memSizeGrid);

	queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, memSizeGrid, h_grid);
	queue.enqueueWriteBuffer(d_auxGrid, CL_FALSE, 0, memSizeGrid, h_auxGrid);


	size_t reqBlocks = gridWidth * gridHeight / numThreads;
	unsigned short blocksCount = (unsigned short)min((size_t)32768, reqBlocks);
	for (size_t i = 0; i < iterations; i++) {
		// Set the kernel arguments
		gameOfLife_kernel.setArg(0, d_grid);
		gameOfLife_kernel.setArg(1, d_auxGrid);
		gameOfLife_kernel.setArg(2, gridWidth);
		gameOfLife_kernel.setArg(3, gridHeight);

		// Execute the kernel
		cl::NDRange global(blocksCount);
		cl::NDRange local(numThreads);
		queue.enqueueNDRangeKernel(gameOfLife_kernel, cl::NullRange, global, local);
		std::swap(d_grid, d_auxGrid);
	}
	// Copy the output data back to the host
	queue.enqueueReadBuffer(d_grid, CL_TRUE, 0, blocksCount, h_grid);
}


void printGrid(char* grid, size_t width, size_t height) {
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


void runGame() {
	initGrid(25, 25, false);
	while (true) {
		if (system("CLS")) {
			system("clear");
		}
		std::cout << "[OpenCL Game Of Life]" << std::endl;
		iterateGPU(1, 32);
		printGrid();
		Sleep(800);
	}
}


int main(int argc, char** argv) {
	std::cout << "[OpenCL Game Of Life] - Starting..." << std::endl;
	int error = initializeOpenCL();
	if (error) {
		return 1;
	}

	std::cout << "Inicializado correctamente :D" << std::endl;
	//runTests();
	//runBenchmarks(std::pow(2, 5), std::pow(2, 15)); // Uncomment to measure performance
	Sleep(5000);
	runGame();
	return 0;
}