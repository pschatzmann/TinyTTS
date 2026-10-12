// Test double for the CVITEK TPU runtime: the subset of
// https://github.com/milkv-duo/tpu-sdk-sg200x/blob/main/include/cviruntime.h
// that src/TinyTTS/CviModel.h uses, same names and signatures. The
// implementation is in test/test_cvi.cpp. Lets the TPU wrappers run on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CVI_FMT_FP32 = 0,
  CVI_FMT_INT32 = 1,
  CVI_FMT_UINT32 = 2,
  CVI_FMT_BF16 = 3,
  CVI_FMT_INT16 = 4,
  CVI_FMT_UINT16 = 5,
  CVI_FMT_INT8 = 6,
  CVI_FMT_UINT8 = 7
} CVI_FMT;

#define CVI_DIM_MAX (6)
typedef struct {
  int32_t dim[CVI_DIM_MAX];
  size_t dim_size;
} CVI_SHAPE;

typedef struct {
  char *name;
  CVI_SHAPE shape;
  CVI_FMT fmt;
  size_t count;
  size_t mem_size;
  uint8_t *sys_mem;
} CVI_TENSOR;

typedef void *CVI_MODEL_HANDLE;
typedef int CVI_RC;
#define CVI_RC_SUCCESS 0
#define CVI_RC_FAILURE 0xFFFFFFFF

CVI_RC CVI_NN_RegisterModel(const char *model_file, CVI_MODEL_HANDLE *model);
CVI_RC CVI_NN_GetInputOutputTensors(CVI_MODEL_HANDLE model, CVI_TENSOR **inputs, int32_t *input_num,
                                    CVI_TENSOR **outputs, int32_t *output_num);
CVI_RC CVI_NN_Forward(CVI_MODEL_HANDLE model, CVI_TENSOR inputs[], int32_t input_num, CVI_TENSOR outputs[],
                      int32_t output_num);
CVI_RC CVI_NN_CleanupModel(CVI_MODEL_HANDLE model);
CVI_TENSOR *CVI_NN_GetTensorByName(const char *name, CVI_TENSOR *tensors, int32_t num);
char *CVI_NN_TensorName(CVI_TENSOR *tensor);
void *CVI_NN_TensorPtr(CVI_TENSOR *tensor);
size_t CVI_NN_TensorSize(CVI_TENSOR *tensor);
size_t CVI_NN_TensorCount(CVI_TENSOR *tensor);
CVI_SHAPE CVI_NN_TensorShape(CVI_TENSOR *tensor);

#ifdef __cplusplus
}
#endif
