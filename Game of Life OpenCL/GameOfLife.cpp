#include <iostream>
#include <random>
#include <fstream>
#include <exception>



#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

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
cl::Kernel gameOfLife_if_kernel;

void clear_screen() {
	char fill = ' ';
	COORD tl = { 0,0 };
	CONSOLE_SCREEN_BUFFER_INFO s;
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(console, &s);
	DWORD written, cells = s.dwSize.X * s.dwSize.Y;
	FillConsoleOutputCharacter(console, fill, cells, tl, &written);
	FillConsoleOutputAttribute(console, s.wAttributes, cells, tl, &written);
	SetConsoleCursorPosition(console, tl);
}


int initializeOpenCL() {
	cl::Platform::get(&platforms);

	// Select the platform.
	platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	std::cout << devices[0].getInfo<CL_DEVICE_VENDOR>() << " - " << devices[0].getInfo<CL_DEVICE_VERSION>() << std::endl;

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
	gameOfLife_if_kernel = cl::Kernel(program, "gameOfLifeIfKernel");
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


void iterateGPU(unsigned short numThreads) {
	size_t memSizeGrid = sizeof(char) * gridSize;

	cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_ONLY, memSizeGrid);
	cl::Buffer d_auxGrid = cl::Buffer(context, CL_MEM_WRITE_ONLY, memSizeGrid);

	queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, memSizeGrid, h_grid);
	//queue.enqueueWriteBuffer(d_auxGrid, CL_FALSE, 0, memSizeGrid, h_auxGrid);


	size_t blocksCount = (gridWidth * gridHeight) / numThreads;
	cl::NDRange global(blocksCount * numThreads);
	cl::NDRange local(numThreads);

	// Set the kernel arguments
	gameOfLife_kernel.setArg(0, d_grid);
	gameOfLife_kernel.setArg(1, d_auxGrid);
	gameOfLife_kernel.setArg(2, gridWidth);
	gameOfLife_kernel.setArg(3, gridHeight);

	// Execute the kernel

	queue.enqueueNDRangeKernel(gameOfLife_kernel, cl::NullRange, global, local);

	// Copy the output data back to the host
	queue.enqueueReadBuffer(d_auxGrid, CL_TRUE, 0, memSizeGrid, h_grid);
}

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

void iterateGPU_IF(unsigned short numThreads) {
	size_t memSizeGrid = sizeof(char) * gridSize;

	cl::Buffer d_grid = cl::Buffer(context, CL_MEM_READ_ONLY, memSizeGrid);
	cl::Buffer d_auxGrid = cl::Buffer(context, CL_MEM_WRITE_ONLY, memSizeGrid);

	queue.enqueueWriteBuffer(d_grid, CL_FALSE, 0, memSizeGrid, h_grid);
	//queue.enqueueWriteBuffer(d_auxGrid, CL_FALSE, 0, memSizeGrid, h_auxGrid);


	size_t blocksCount = (gridWidth * gridHeight) / numThreads;
	cl::NDRange global(blocksCount * numThreads);
	cl::NDRange local(numThreads);

	// Set the kernel arguments
	gameOfLife_if_kernel.setArg(0, d_grid);
	gameOfLife_if_kernel.setArg(1, d_auxGrid);
	gameOfLife_if_kernel.setArg(2, gridWidth);
	gameOfLife_if_kernel.setArg(3, gridHeight);

	// Execute the kernel

	queue.enqueueNDRangeKernel(gameOfLife_if_kernel, cl::NullRange, global, local);

	// Copy the output data back to the host
	queue.enqueueReadBuffer(d_auxGrid, CL_TRUE, 0, memSizeGrid, h_grid);
}


void printGrid(char* grid, size_t width, size_t height) {
	std::string s = "";
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++) {
			if (grid[row * width + col] == 1)
				s += " O ";
			else
				s += "   ";
		}
		s += "\n";
	}
	std::cout << s;
}



/// Prints the game of life grid to console.
void printGrid() {
	printGrid(h_grid, gridWidth, gridHeight);
}


void runGame() {
	initGrid(26, 36, false);
	while (true) {
		clear_screen();
		std::cout << "[OpenCL Game Of Life]" << std::endl;
		iterateGPU(64);
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
	Sleep(2000);
	runGame();
	return 0;
}