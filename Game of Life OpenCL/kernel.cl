
kernel void gameOfLifeKernel(global char* golGrid, global char* auxGolGrid, ulong width, ulong height) {

	unsigned int size = width * height;

	int gid = get_global_id(0);			// 0 .. total_array_size-1
	int gsize = get_global_size(0);		// total_array_size

	for (unsigned int cellId = gid; cellId < size; cellId += gsize) {

		unsigned int x = cellId % width;
		unsigned int y = cellId - x;
		unsigned int xLeft = (x + width - 1) % width;
		unsigned int xRight = (x + 1) % width;
		unsigned int yUp = (y + size - width) % size;
		unsigned int yDown = (y + width) % size;

		unsigned int aliveCells = golGrid[xLeft + yUp] + golGrid[x + yUp]
			+ golGrid[xRight + yUp] + golGrid[xLeft + y] + golGrid[xRight + y]
			+ golGrid[xLeft + yDown] + golGrid[x + yDown] + golGrid[xRight + yDown];

		auxGolGrid[x + y] = aliveCells == 3 || ((aliveCells == 2 && golGrid[x + y]) ? 1 : 0);
	}
}

// Kernel that computes an iteration of the game of life using ifs to check for
// alive neighbours.
__global__ void gameOfLifeIfKernel(char *golGrid,
	unsigned int width, unsigned int height, char *auxGolGrid) {

	unsigned int size = width * height;

	int gid = get_global_id(0);			// 0 .. total_array_size-1
	int gsize = get_global_size(0);		// total_array_size

	for (unsigned int cellId = gid; cellId < size; cellId += gsize) {
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