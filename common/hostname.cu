// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#communicator-creation-and-destruction-examples
#pragma once
#include <stdint.h>
#include <unistd.h>

// Convert a hostname into a unique hash
static uint64_t getHostHash(const char *string)
{
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

// Retrieve the hostname of a given MPI process
static void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}