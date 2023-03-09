#include <iomanip>
#include <iostream>

#include "mpi.h"
#include "nccl.h"

#include "../common/error-catch.cpp"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"
#include "../common/synchronize.cu"

int main(int argc, char **argv)
{
    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));

    // Set MPI size and rank
    int size;
    int rank;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    std::cout << std::setiosflags(std::ios_base::fixed);

    int device_size;
    CUDACHECK(cudaGetDeviceCount(&device_size));
    if (rank == 0)
    {
        std::cout << "nccl-pingpong" << std::endl;
        std::cout << "CUDA devices available: " << device_size << std::endl;
        std::cout << "MPI world size: " << size << std::endl;
    }

    // Figure out what host the current MPI process is running on
    uint64_t hostHashs[size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    // Compute and set the local rank based on the hostname
    int local_rank = 0;
    for (int i = 0; i < size; i++)
    {
        if (i == rank)
        {
            break;
        }
        if (hostHashs[i] == hostHashs[rank])
        {
            local_rank++;
        }
    }

    // Initialize a unique NCCL ID at process 0 and broadcast it to all others
    ncclUniqueId id;
    if (rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL
    ncclComm_t comm;
    cudaStream_t stream;
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaStreamCreate(&stream));
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    // Host variables
    int *h_send_data;
    int *h_recv_data;

    // Device variables
    int *d_send_data;
    int *d_recv_data;

    int ite_count = 1;

    // Warm up
    {
        h_send_data = new int[4];
        h_recv_data = new int[4];
        int bytes = 4 * sizeof(int);
        std::fill_n(h_send_data, 4, rank);
        std::fill_n(h_recv_data, 4, -1);

        // Print send data
        // if (rank == 0)
        // {
        //     std::cout << "Rank " << rank << " send data:\t[";
        //     for (int i = 0; i < 4; i++)
        //     {
        //         std::cout << " " << h_send_data[i] << " ";
        //     }
        //     std::cout << "]" << std::endl;
        // }

        // if (rank == size - 1)
        // {
        //     std::cout << "Rank " << rank << " recieve data:\t[";
        //     for (int i = 0; i < 4; i++)
        //     {
        //         std::cout << " " << h_recv_data[i] << " ";
        //     }
        //     std::cout << "]" << std::endl;
        // }

        CUDACHECK(cudaMalloc((void **)&d_send_data, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recv_data, bytes));
        CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv_data, 0, bytes));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            ncclGroupStart();
            if (rank == size - 1)
            {
                NCCLCHECK(ncclRecv((char *)d_recv_data, 4, ncclInt, 0, comm, stream));
            }
            if (rank == 0)
            {
                NCCLCHECK(ncclSend((char *)d_send_data, 4, ncclInt, size - 1, comm, stream));
            }
            ncclGroupEnd();
            ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();
            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, size - 1, MPI_COMM_WORLD));

            // Verify the recieved data
            if (rank == size - 1)
            {
                cudaMemcpy(h_recv_data, d_recv_data, bytes, cudaMemcpyDeviceToHost);

                bool valid = true;
                for (int i = 0; i < 4; i++)
                {
                    if (h_recv_data[i] != 0)
                    {
                        valid = false;
                        break;
                    }
                }

                if (valid)
                {
                    std::cout << "\nWarm up (success): " << 4 << " elements sent in " << elapsedTime << " seconds" << std::endl;
                }
                else
                {
                    std::cout << "(failed) rank " << rank << " recieved data:\t[";
                    for (int i = 0; i < 4; i++)
                    {
                        std::cout << " " << h_recv_data[i] << " ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }

        // Free all allocated memory
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        delete[] h_send_data;
        delete[] h_recv_data;
        CUDACHECK(cudaFree(d_send_data));
        CUDACHECK(cudaFree(d_recv_data));
    }

    // Actual test
    for (int count = 4; count <= 4096; count *= 2)
    {
        h_send_data = new int[count];
        h_recv_data = new int[count];
        int bytes = count * sizeof(int);
        std::fill_n(h_send_data, count, rank);
        std::fill_n(h_recv_data, count, -1);

        // Print send data
        // if (rank == 0)
        // {
        //     std::cout << "Rank " << rank << " send data:\t[";
        //     for (int i = 0; i < count; i++)
        //     {
        //         std::cout << " " << h_send_data[i] << " ";
        //     }
        //     std::cout << "]" << std::endl;
        // }

        // if (rank == size - 1)
        // {
        //     std::cout << "Rank " << rank << " recieve data:\t[";
        //     for (int i = 0; i < count; i++)
        //     {
        //         std::cout << " " << h_recv_data[i] << " ";
        //     }
        //     std::cout << "]" << std::endl;
        // }

        CUDACHECK(cudaMalloc((void **)&d_send_data, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recv_data, bytes));
        CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv_data, 0, bytes));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            if (rank == size - 1)
            {
                NCCLCHECK(ncclRecv((char *)d_recv_data, count, ncclInt, 0, comm, stream));
            }
            if (rank == 0)
            {
                NCCLCHECK(ncclSend((char *)d_send_data, count, ncclInt, size - 1, comm, stream));
            }
            // ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();
            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, size - 1, MPI_COMM_WORLD));

            // Verify the recieved data
            if (rank == size - 1)
            {
                cudaMemcpy(h_recv_data, d_recv_data, bytes, cudaMemcpyDeviceToHost);

                bool valid = true;
                for (int i = 0; i < 4; i++)
                {
                    if (h_recv_data[i] != 0)
                    {
                        valid = false;
                        break;
                    }
                }

                if (valid)
                {
                    std::cout << "\n(success): " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
                }
                else
                {
                    std::cout << "(failed) rank " << rank << " recieved data:\t[";
                    for (int i = 0; i < count; i++)
                    {
                        std::cout << " " << h_recv_data[i] << " ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        CUDACHECK(cudaFree(d_send_data));
        CUDACHECK(cudaFree(d_recv_data));
        d_send_data = nullptr;
        d_recv_data = nullptr;
    }

    // Destroy NCCL communicator
    ncclCommDestroy(comm);

    // Finalize MPI
    MPICHECK(MPI_Finalize());
    return 0;
}