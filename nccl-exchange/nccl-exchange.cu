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
        std::cout << "nccl-exchange" << std::endl;
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
    char *h_send_data;
    char *h_recv_data;

    // Device variables
    char *d_send_data;
    char *d_recv_data;

    int ite_count = 1;
    int loopCount = std::ceil(std::log2(size));

    // Warm up
    {
        int count = 32;
        int bytes = count * sizeof(char);
        h_send_data = new char[count];
        h_recv_data = new char[count];
        std::fill_n(h_send_data, count, rank);
        std::fill_n(h_recv_data, count, -1);

        CUDACHECK(cudaMalloc((void **)&d_send_data, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recv_data, bytes));
        CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv_data, 0, bytes));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            int distance = std::pow(2, loopCount - 1);
            for (int i = 0; i < loopCount; i++)
            {
                int sendrank = (rank + distance) % size;
                int recvrank = (rank - distance + size) % size;
                ncclGroupStart();
                NCCLCHECK(ncclSend((char *)d_send_data, count, ncclChar, sendrank, comm, stream));
                NCCLCHECK(ncclRecv((char *)d_recv_data, count, ncclChar, recvrank, comm, stream));
                ncclGroupEnd();
                distance /= 2;
            }
            ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();

            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (rank == 0)
            {
                std::cout << "Warm up descending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
            }

            // cudaMemcpy(h_recv_data, d_recv_data, bytes, cudaMemcpyDeviceToHost);
            // std::cout << "Rank " << rank << " recieved data:\t[";
            // for (int i = 0; i < count; i++)
            // {
            //     std::cout << " " << h_recv_data[i] << " ";
            // }
            // std::cout << "]" << std::endl;
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            int distance = 1;
            for (int i = 0; i < loopCount; i++)
            {
                int sendrank = (rank + distance) % size;
                int recvrank = (rank - distance + size) % size;
                ncclGroupStart();
                NCCLCHECK(ncclSend((char *)d_send_data, count, ncclChar, sendrank, comm, stream));
                NCCLCHECK(ncclRecv((char *)d_recv_data, count, ncclChar, recvrank, comm, stream));
                ncclGroupEnd();
                distance *= 2;
            }
            ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();

            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (rank == 0)
            {
                std::cout << "Warm up ascending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
            }

            // cudaMemcpy(h_recv_data, d_recv_data, bytes, cudaMemcpyDeviceToHost);
            // std::cout << "Rank " << rank << " recieved data:\t[";
            // for (int i = 0; i < count; i++)
            // {
            //     std::cout << " " << h_recv_data[i] << " ";
            // }
            // std::cout << "]" << std::endl;
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        delete[] h_send_data;
        delete[] h_recv_data;
        CUDACHECK(cudaFree(d_send_data));
        CUDACHECK(cudaFree(d_recv_data));
    }

    // Actual test
    for (int count = 32; count <= 8192; count *= 2)
    {
        int bytes = count * sizeof(char);
        h_send_data = new char[count];
        h_recv_data = new char[count];
        std::fill_n(h_send_data, count, rank);
        std::fill_n(h_recv_data, count, -1);

        CUDACHECK(cudaMalloc((void **)&d_send_data, bytes));
        CUDACHECK(cudaMalloc((void **)&d_recv_data, bytes));
        CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv_data, 0, bytes));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            int distance = std::pow(2, loopCount - 1);
            for (int i = 0; i < loopCount; i++)
            {
                int sendrank = (rank + distance) % size;
                int recvrank = (rank - distance + size) % size;
                ncclGroupStart();
                NCCLCHECK(ncclSend((char *)d_send_data, count, ncclChar, sendrank, comm, stream));
                NCCLCHECK(ncclRecv((char *)d_recv_data, count, ncclChar, recvrank, comm, stream));
                ncclGroupEnd();
                distance /= 2;
            }
            ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();

            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (rank == 0)
            {
                std::cout << "descending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
            }
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        for (int i = 0; i < ite_count; i++)
        {
            double start = MPI_Wtime();
            int distance = 1;
            for (int i = 0; i < loopCount; i++)
            {
                int sendrank = (rank + distance) % size;
                int recvrank = (rank - distance + size) % size;
                ncclGroupStart();
                NCCLCHECK(ncclSend((char *)d_send_data, count, ncclChar, sendrank, comm, stream));
                NCCLCHECK(ncclRecv((char *)d_recv_data, count, ncclChar, recvrank, comm, stream));
                ncclGroupEnd();
                distance *= 2;
            }
            ncclStreamSynchronize(stream, comm);
            double stop = MPI_Wtime();

            const double localElapsedTime = stop - start;
            double elapsedTime;
            MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
            if (rank == 0)
            {
                std::cout << "ascending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
            }
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        delete[] h_send_data;
        delete[] h_recv_data;
        CUDACHECK(cudaFree(d_send_data));
        CUDACHECK(cudaFree(d_recv_data));
    }

    MPI_Finalize();
    return 0;
}