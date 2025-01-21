#include "pch.h"
#include "NeuralNetwork.h"

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // Initialize global variables or resources here.
        // Example: Initialize any global neural network or other resources
        // g_neuralNetwork = new NeuralNetwork(3, 5, 2); // Example global NN initialization
        break;

    case DLL_THREAD_ATTACH:
        // Called when a new thread is created in the process that loaded the DLL
        // Typically used for thread-specific initialization (e.g., allocating thread-local storage).
        break;

    case DLL_THREAD_DETACH:
        // Called when a thread exits in the process that loaded the DLL.
        // Cleanup thread-specific resources (e.g., freeing memory allocated for the thread).
        break;

    case DLL_PROCESS_DETACH:
        // Cleanup global resources when the DLL is unloaded
        // Example: Free memory, close files, etc.
        // delete g_neuralNetwork; // If you have a global object like this.
        break;
    }
    return TRUE;
}