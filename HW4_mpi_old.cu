#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <mpi.h>

#define DEV_NO 0
#define ROUND_MAX 4

const int INF = 1000000000;
const int V = 20010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
bool cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);
__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu);
__global__ void p1_cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch_int);
__global__ void p2_cal_kernel(int B, int Round, int n, int* Dist_gpu, int pitch_int);
/*MPI*/
__global__ void p3_cal_kernel(int B, int Round, int k_i, int n, int* Dist_gpu, int pitch_int);


int n, m;	// Number of vertices, edges
int* Dist;
int* Dist_gpu;
cudaDeviceProp prop;
size_t pitch;

/*MPI*/
int num_gpus;
int rank, size;
int chunk, more;
int proc_start_b_i, proc_end_b_i;
int valid_row;


int main(int argc, char* argv[])
{	
	assert(argc==4);
	cudaGetDeviceProperties(&prop, DEV_NO);
    cudaGetDeviceCount(&num_gpus);

	input(argv[1]);
	int B = atoi(argv[3]);
	assert((B*B-1)/prop.maxThreadsPerBlock < ROUND_MAX);
    /*MPI*/
    int Round = ceil(n, B);

    /*MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    chunk = Round / size;
    more = Round % size;
    proc_start_b_i = (rank<more)? (chunk+1)*rank : (chunk+1)*more + chunk*(rank-more);
    proc_end_b_i = (rank<more)? proc_start_b_i + (chunk+1) : proc_start_b_i + chunk;
    valid_row = (proc_end_b_i*B > n)? n - proc_start_b_i*B : (proc_end_b_i - proc_start_b_i)*B;
    printf("%d: %d %d\n", rank, proc_start_b_i, proc_end_b_i);
    
	//auto start = std::chrono::high_resolution_clock::now();
	block_FW(B);
	//auto end = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double> diff = end - start;
    //std::cout << rank << ": " << diff.count() * 1000 << "(ms)\n";
    //MPI_Barrier(MPI_COMM_WORLD);
    
    int rc;
    MPI_File fh;
    MPI_Status st;
    rc = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_File_write_at(fh, sizeof(int)*proc_start_b_i*B*n, Dist*proc_start_b_i*B*n, valid_row*n, MPI_INT, &st);
    MPI_File_write_at(fh, sizeof(int)*proc_start_b_i*B*n, Dist+proc_start_b_i*B*n, valid_row*n, MPI_INT, &st);
    MPI_File_close(&fh);
    //MPI_Finalize();
	//output(argv[2]);

	return 0;
}

void input(char *inFileName)
{
    FILE *infile = fopen(inFileName, "r");
    fseek(infile, 0, SEEK_END);
    long lsize = ftell(infile);
    rewind(infile);

    char* input_buff = (char*) malloc(sizeof(char)*lsize);
    assert(input_buff != NULL);
    size_t result = fread(input_buff, 1, lsize, infile);
    assert(result == lsize);
    n = atoi(strtok(input_buff, " \n"));
    m = atoi(strtok(NULL, " \n"));


    //Dist = (int*)malloc(n*n*sizeof(int));
    cudaMallocHost(&Dist, sizeof(int)*n*n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) Dist[i*n + j] = 0;
            else        Dist[i*n + j] = INF;
        }
    }
    //printf("%d %d\n", n, m);

    while (--m >= 0) {
        int a, b, v;
        a = atoi(strtok(NULL, " \n"));
        b = atoi(strtok(NULL, " \n"));
        v = atoi(strtok(NULL, " \n"));
        Dist[a*n + b] = v;
    }
    fclose(infile);
    free(input_buff);
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	fwrite(Dist, sizeof(int), n*n, outfile);
    fclose(outfile);
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B)
{
    /*MPI*/
    unsigned int g_id = rank % num_gpus;
    cudaSetDevice(g_id); 

	int num_thread = (B*B>prop.maxThreadsPerBlock)? prop.maxThreadsPerBlock: B*B; 
	int round = ceil(n, B);
	cudaMallocPitch((void**)&Dist_gpu, &pitch,n*sizeof(int), n+128);    /*MPI*/
	int pitch_int = pitch / sizeof(int);
	cudaMemcpy2D(Dist_gpu, pitch, Dist, n*sizeof(int), n*sizeof(int), n, cudaMemcpyHostToDevice);

    /*MPI*/
    //MPI_Status st;
    MPI_Request req;
    int sr_buff[n*B];
    dim3 grid3_p(proc_end_b_i-proc_start_b_i, round-1);	

	dim3 grid2(round-1, 2);
	dim3 grid3(round-1, round-1);
	dim3 block(B, num_thread/B);
	for (int r = 0; r < round; ++r) {
        //printf("%d %d\n", r, round);
		/* Phase 1*/
		p1_cal_kernel<<< 1, block, B*B*sizeof(int) >>>(B, r,	r,	r,	1,	1, n, Dist_gpu, pitch_int);

		/* Phase 2*/
		p2_cal_kernel<<< grid2, block, 2*B*B*sizeof(int) >>>(B, r, n, Dist_gpu, pitch_int); 
		// cal_kernel<<<           r*1, num_thread >>>(B, r,     r,     0,             r,             1, n, Dist_gpu);
		// cal_kernel<<< (round-r-1)*1, num_thread >>>(B, r,     r,  r +1,  round - r -1,             1, n, Dist_gpu);
		// cal_kernel<<<           1*r, num_thread >>>(B, r,     0,     r,			 1, 			r, n, Dist_gpu);
		// cal_kernel<<< 1*(round-r-1), num_thread >>>(B, r,  r +1,     r,             1,  round - r -1, n, Dist_gpu);

		/* Phase 3*/
			
		//p3_cal_kernel<<< grid3_p, block>>>(B, r, proc_start_b_i, n, Dist_gpu, pitch_int);
		p3_cal_kernel<<< grid3, block>>>(B, r, 0, n, Dist_gpu, pitch_int);
        
        /*MPI*/
        //int more = round%size;
        /*if(r+1 < round) {
        int sender_rank = (r+1 < more*(chunk+1))? (r+1)/(chunk+1) :  more + (r+1-more*(chunk+1))/chunk;
        //printf("%d %d %d rank:%d sender_rank:%d\n", more, chunk, r, rank, sender_rank);
        if(rank == sender_rank) cudaMemcpy2D(sr_buff, n*sizeof(int), Dist_gpu+(r+1)*B*pitch_int, pitch, n*sizeof(int), B, cudaMemcpyDeviceToHost); 
        MPI_Bcast(sr_buff, n*B, MPI_INT, sender_rank, MPI_COMM_WORLD);
        if(rank != sender_rank) cudaMemcpy2D(Dist_gpu+(r+1)*B*pitch_int, pitch, sr_buff, n*sizeof(int), n*sizeof(int), B, cudaMemcpyHostToDevice);
        }*/

		// cal_kernel<<<                     r*r, num_thread >>>(B, r,     0,     0,            r,             r, n, Dist_gpu);
		// cal_kernel<<<           (round-r-1)*r, num_thread >>>(B, r,     0,  r +1,  round -r -1,             r, n, Dist_gpu);
		// cal_kernel<<<           r*(round-r-1), num_thread >>>(B, r,  r +1,     0,            r,  round - r -1, n, Dist_gpu);
		// cal_kernel<<< (round-r-1)*(round-r-1), num_thread >>>(B, r,  r +1,  r +1,  round -r -1,  round - r -1, n, Dist_gpu);
		
	}
    
    /*MPI*/
    //printf("%d, %d", Dist, Dist_gpu);
	cudaMemcpy2D(Dist+proc_start_b_i*B*n, n*sizeof(int), Dist_gpu+proc_start_b_i*B*pitch_int, pitch, n*sizeof(int), valid_row, cudaMemcpyDeviceToHost);
}

__global__ void cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu) {
	
	int b_i = block_start_x + blockIdx.x / block_width;
	int b_j = block_start_y + blockIdx.x % block_width;
	
	int inner_round = (B*B-1)/blockDim.x + 1;
	
	//__shared__ int shared_mem = 
	
	for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {

		for(int r=0; r<inner_round; r++){

			int i = b_i * B + (threadIdx.x + r*blockDim.x) / B;
			int j = b_j * B + (threadIdx.x + r*blockDim.x) % B;

			if ((i>=n) | (j>=n)) continue ;
			//if ((Dist_gpu[i*n+k] + Dist_gpu[k*n+j])==73) printf("%d, %d, %d, %d\n", i, j, k, n);
			if (Dist_gpu[i*n+k] + Dist_gpu[k*n+j] < Dist_gpu[i*n+j]) {
				Dist_gpu[i*n+j] = Dist_gpu[i*n+k] + Dist_gpu[k*n+j];
			}
		}
		__syncthreads();
	}
	
}

__global__ void p1_cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch_int) {
	
	// if(blockIdx.x==0 && threadIdx.x==0 && Round==0){
	// 	for(int i =0; i<n; i++){
	// 		for(int j=0; j<n; j++){
	// 			printf("%d ",Dist_gpu[i*pitch_int+j]);
	// 		}
	// 		printf("\n");
	// 	}
	// }

	int b_i = block_start_x ;
	int b_j = block_start_y ;
	
	//int inner_round = (B*B-1)/blockDim.x + 1;
	
	extern __shared__ int shared_mem[]; 
	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];
	
	#pragma unroll
	for(int r=0; r<4; r++){
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		//if(inner_i[r]>=B) continue;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		if (!((global_i[r]>=n) | (global_j[r]>=n))) 
			shared_mem[inner_i[r]*B + inner_j[r]] = Dist_gpu[global_i[r]*pitch_int + global_j[r]]; 		
	}


	// if(blockIdx.x==0 && threadIdx.x==0){
	// 	for(int i=0; i<B; i++){
	// 		for(int j=0; j<B; j++){
	// 			printf("%d ", shared_mem[i*B+j]);
	// 		}
	// 		printf("\n");
	// 	}
	// }
	// __syncthreads();

	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		__syncthreads();

		#pragma unroll
		for(int r=0; r<4; r++){
			//if(inner_i[r]>=B) continue;
			if ((global_i[r]>=n) | (global_j[r]>=n)) continue ;			

			if (shared_mem[inner_i[r]*B+inner_j[r]] > shared_mem[inner_i[r]*B+k] + shared_mem[k*B+inner_j[r]]) {
				shared_mem[inner_i[r]*B+inner_j[r]] = shared_mem[inner_i[r]*B+k] + shared_mem[k*B+inner_j[r]];
			}
		}
		
	}

	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		if (!((global_i[r]>=n) | (global_j[r]>=n))) 
			Dist_gpu[global_i[r]*pitch_int + global_j[r]] = shared_mem[inner_i[r]*B + inner_j[r]];	
	}
	
}


extern __shared__ int shared_mem[]; 
__global__ void p2_cal_kernel(int B, int Round, int n, int* Dist_gpu, int pitch_int) {
	
	int b_i, b_j;
	if(blockIdx.y==0){
		b_i = Round;
		b_j = blockIdx.x + (blockIdx.x>=Round);
	}
	else{
		b_i = blockIdx.x + (blockIdx.x>=Round);
		b_j = Round;
	}
	
	//int inner_round = (B*B-1)/blockDim.x + 1;
	
	
	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];
	
	#pragma unroll
	for(int r=0; r<4; r++){
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		//if(inner_i[r]>=B) continue;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		int global_pivot_i = Round * B + inner_i[r];
		int global_pivot_j = Round * B + inner_j[r];
		if (!((global_i[r]>=n) | (global_j[r]>=n))) 
			shared_mem[inner_i[r]*B + inner_j[r]] = Dist_gpu[global_i[r]*pitch_int + global_j[r]];
		if (!((global_pivot_i>=n) | (global_pivot_j>=n))) 
			shared_mem[inner_i[r]*B + inner_j[r] + B*B] = Dist_gpu[global_pivot_i*pitch_int + global_pivot_j];
	}
	

	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		__syncthreads();

		#pragma unroll
		for(int r=0; r<4; r++){
			//if(inner_i[r]>=B) continue;
			if ((global_i[r]>=n) | (global_j[r]>=n)) continue ;

			//if ((Dist_gpu[i*n+k] + Dist_gpu[k*n+j])==73) printf("%d, %d, %d, %d\n", i, j, k, n);
			if (shared_mem[inner_i[r]*B+inner_j[r]] > shared_mem[inner_i[r]*B+k + !blockIdx.y*B*B] + shared_mem[k*B+inner_j[r] + blockIdx.y*B*B]) {
				shared_mem[inner_i[r]*B+inner_j[r]] = shared_mem[inner_i[r]*B+k + !blockIdx.y*B*B] + shared_mem[k*B+inner_j[r] + blockIdx.y*B*B];
			}
			
		}
		
	}
	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		if (!((global_i[r]>=n) | (global_j[r]>=n))) 
			Dist_gpu[global_i[r]*pitch_int + global_j[r]] = shared_mem[inner_i[r]*B + inner_j[r]];
				
	}

	
}

__global__ void p3_cal_kernel(int B, int Round, int k_i, int n, int* Dist_gpu, int pitch_int) {

    /*MPI*/
	int b_i = k_i+blockIdx.x + ((k_i+blockIdx.x)>=Round);
	int b_j = blockIdx.y + (blockIdx.y>=Round);

	__shared__ int shared_mem[8192]; 
	//int inner_round = (B*B-1)/blockDim.x + 1;
		
	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];
	int my_dist[ROUND_MAX];
	
	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		int row_pivot_i = global_i[r];
		int row_pivot_j = Round * B + inner_j[r];
		int col_pivot_i = Round * B + inner_i[r];
		int col_pivot_j = global_j[r];

		my_dist[r] = Dist_gpu[global_i[r]*pitch_int + global_j[r]];
		shared_mem[inner_i[r]*B + inner_j[r] ] = Dist_gpu[row_pivot_i*pitch_int + row_pivot_j];
		shared_mem[inner_i[r]*B + inner_j[r] + B*B] = Dist_gpu[col_pivot_i*pitch_int + col_pivot_j];
		
	}

	__syncthreads();
	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		#pragma unroll
		for(int r=0; r<4; r++){			
			int tmp = shared_mem[inner_i[r]*B+k ] + shared_mem[k*B+inner_j[r] +B*B];
			if (my_dist[r] > tmp) {
				my_dist[r] = tmp;
			}			
		}
	}

	#pragma unroll
	for(int r=0; r<4; r++){
		Dist_gpu[global_i[r]*pitch_int + global_j[r]] = my_dist[r];
		 		
	}

}
