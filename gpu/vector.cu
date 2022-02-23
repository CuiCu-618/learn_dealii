
#include <deal.II/base/timer.h>

#include <deal.II/lac/cuda_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector_operation.h>

#include <iostream>

using namespace dealii;

__global__ 
void test_print()
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("[%d,%d] : %d\n", blockIdx.x, threadIdx.x, index);
}

int main()
{
    using number = float;
    static constexpr unsigned int size = 1000000;
    Vector<number> vec(size);
    {
        Timer timer;
        for (unsigned int i=0; i<1000; ++i)
            vec.add(1.);
        timer.stop();
        std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
        std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";       
    }

    LinearAlgebra::ReadWriteVector<number> rw_vector(size);
    LinearAlgebra::CUDAWrappers::Vector<number> vec_dev(size);
    
    {
        Timer timer;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        vec_dev.import(rw_vector, VectorOperation::insert);
        for (unsigned int i=0; i<1000; ++i)
            vec_dev.add(1.); 
        rw_vector.import(vec_dev, VectorOperation::insert);
        cudaEventRecord(stop);
        timer.stop();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds.\n";
        std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";  
        std::cout << "cudaEventElapsedTime: " << milliseconds << " milliseconds.\n";  
    }

    test_print<<<2,32>>>();

    return 0;
}