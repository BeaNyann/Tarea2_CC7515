kernel void gameOfLifeKernel(global char* golGrid, global char* auxGolGrid, uint width, uint height) {

	unsigned int size = width * height;


	int gid = get_global_id(0);			// 0 .. total_array_size-1
	//int numGlobal = get_global_size(0);
	int numItems = get_local_size(0);	// # work-items per work-group
	int tnum = get_local_id(0);			// thread (i.e., work-item) number in this work-group
										// 0 .. numItems-1
	int wgNum = get_group_id(0);		// which work-group number this is in
	/*
	for (unsigned int cellId = wgNum * numItems + tnum; cellId < size; cellId += numItems * numGlobal) {
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
	*/
}
