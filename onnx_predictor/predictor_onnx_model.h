#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>


struct OnnxSession {
    Ort::Session session;
    Ort::MemoryInfo memoryInfo;
    Ort::AllocatorWithDefaultOptions allocator;
};

class OnnxModel {
  public:
    OnnxModel();
    ~OnnxModel();

    void run(const std::vector<Ort::Value>& inputTensors,
             std::vector<Ort::Value>& outputTensors);

    // void init(const std::string& model_file_path);
  
  private:
    // Ort::Session session;
    // Ort::AllocatorWithDefaultOptions allocator;
    // Ort::MemoryInfo memoryInfo;

    Ort::Session createSession(const std::string& model_file_path);
    Ort::AllocatorWithDefaultOptions allocateMemory();
    Ort::MemoryInfo getMemoryInfo();
};