// Host test for the Milk-V Duo TPU wrappers (CviTpuVocoder, CviTpuFlow) --
// without a TPU: test/cvi_fake/cviruntime.h declares the CVITEK runtime API,
// and the fake runtime below "runs" a model by calling the CPU Vocoder/Flow on
// the frames its mask marks as valid. That's what the real models compute
// (see research/milkv_duo/), so the wrappers' output must match the plain CPU
// pipeline: this checks their tensor layout, padding, masks and copies.
//
// Fake model files are names, not files: "fake:<kind>:<L>[:<hop>[:<file name>]]"
// with kind vocoder, flow or int8 (a vocoder with an INT8 output, which must
// be rejected). <hop> (output samples per frame) defaults to the CPU
// vocoder's; <file name> stands in for a real cvimodel name, whose sample rate
// checkModel() compares with the weights.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "TinyTTS.h"
#include "TinyTTS/CviTpuFlow.h"
#include "TinyTTS/CviTpuVocoder.h"

using namespace tinytts;

// ---- fake CVITEK runtime ----

static const Vocoder* g_vocoder = nullptr;
static const Flow* g_flow = nullptr;

namespace {
struct FakeTensor {
  std::string name;
  std::vector<int32_t> dims;
  std::vector<float> data;
};
struct FakeModel {
  std::string kind;
  int length = 0;
  int hop = 0;
  std::vector<FakeTensor> in, out;
  std::vector<CVI_TENSOR> cin, cout;
};

FakeTensor tensor(const char* name, int c, int l) { return {name, {1, c, l, 1}, std::vector<float>((size_t)c * l)}; }

void wrap(std::vector<FakeTensor>& ts, std::vector<CVI_TENSOR>& out, CVI_FMT fmt) {
  out.clear();
  for (FakeTensor& t : ts) {
    CVI_TENSOR c{};
    c.name = &t.name[0];
    c.shape.dim_size = t.dims.size();
    for (size_t i = 0; i < t.dims.size(); i++) c.shape.dim[i] = t.dims[i];
    c.fmt = fmt;
    c.count = t.data.size();
    c.mem_size = t.data.size() * sizeof(float);
    c.sys_mem = (uint8_t*)t.data.data();
    out.push_back(c);
  }
}

// [1, C, L] channel-first -> [rows [begin, end), C] Mat.
Mat fromChannelFirst(const float* p, int C, int L, int begin, int end) {
  Mat m(end - begin, C);
  for (int t = begin; t < end; t++)
    for (int c = 0; c < C; c++) m.at(t - begin, c) = p[(size_t)c * L + t];
  return m;
}

// The 1-run of a 0/1 mask, as [begin, end).
void maskRange(const std::vector<float>& mask, int& begin, int& end) {
  begin = 0;
  while (begin < (int)mask.size() && mask[begin] == 0.0f) begin++;
  end = begin;
  while (end < (int)mask.size() && mask[end] != 0.0f) end++;
}
}  // namespace

extern "C" {
CVI_RC CVI_NN_RegisterModel(const char* model_file, CVI_MODEL_HANDLE* model) {
  std::vector<std::string> f;
  std::stringstream ss(model_file);
  for (std::string part; std::getline(ss, part, ':');) f.push_back(part);
  if (f.size() < 3 || f[0] != "fake") return CVI_RC_FAILURE;
  FakeModel* m = new FakeModel();
  m->kind = f[1];
  m->length = std::stoi(f[2]);
  m->hop = f.size() > 3 && std::stoi(f[3]) > 0 ? std::stoi(f[3]) : g_vocoder->hopLength();
  const int L = m->length;
  if (m->kind == "vocoder" || m->kind == "int8") {
    m->in = {tensor("z", 32, L), tensor("g", 128, 1), tensor("mask", 1, L)};
    m->out = {tensor("audio_Tanh_f32", 1, L * m->hop)};
  } else if (m->kind == "flow") {
    m->in = {tensor("z_p", 32, L), tensor("y_mask", 1, L), tensor("g", 128, 1)};
    m->out = {tensor("z_Mul_f32", 32, L)};
  } else {
    delete m;
    return CVI_RC_FAILURE;
  }
  wrap(m->in, m->cin, CVI_FMT_FP32);
  wrap(m->out, m->cout, m->kind == "int8" ? CVI_FMT_INT8 : CVI_FMT_FP32);
  *model = m;
  return CVI_RC_SUCCESS;
}

CVI_RC CVI_NN_GetInputOutputTensors(CVI_MODEL_HANDLE model, CVI_TENSOR** inputs, int32_t* input_num,
                                    CVI_TENSOR** outputs, int32_t* output_num) {
  FakeModel* m = (FakeModel*)model;
  *inputs = m->cin.data();
  *input_num = (int32_t)m->cin.size();
  *outputs = m->cout.data();
  *output_num = (int32_t)m->cout.size();
  return CVI_RC_SUCCESS;
}

CVI_RC CVI_NN_Forward(CVI_MODEL_HANDLE model, CVI_TENSOR*, int32_t, CVI_TENSOR*, int32_t) {
  FakeModel* m = (FakeModel*)model;
  const int L = m->length;
  Mat g = fromChannelFirst(m->in[m->kind == "flow" ? 2 : 1].data.data(), 128, 1, 0, 1);
  std::vector<float>& out = m->out[0].data;
  std::fill(out.begin(), out.end(), 0.0f);
  int vb, ve;
  if (m->kind == "flow") {
    maskRange(m->in[1].data, vb, ve);
    Mat z = g_flow->reverse(fromChannelFirst(m->in[0].data.data(), 32, L, vb, ve), g);
    for (int t = 0; t < z.rows(); t++)
      for (int c = 0; c < 32; c++) out[(size_t)c * L + vb + t] = z.at(t, c);
  } else {
    maskRange(m->in[2].data, vb, ve);
    if (m->hop != g_vocoder->hopLength()) return CVI_RC_FAILURE;  // a model for other weights
    auto audio = g_vocoder->forward(fromChannelFirst(m->in[0].data.data(), 32, L, vb, ve), g);
    std::copy(audio.begin(), audio.end(), out.begin() + (size_t)vb * m->hop);
  }
  return CVI_RC_SUCCESS;
}

CVI_RC CVI_NN_CleanupModel(CVI_MODEL_HANDLE model) {
  delete (FakeModel*)model;
  return CVI_RC_SUCCESS;
}

CVI_TENSOR* CVI_NN_GetTensorByName(const char* name, CVI_TENSOR* tensors, int32_t num) {
  for (int i = 0; i < num; i++)
    if (std::strcmp(tensors[i].name, name) == 0) return &tensors[i];
  return nullptr;
}
char* CVI_NN_TensorName(CVI_TENSOR* tensor) { return tensor->name; }
void* CVI_NN_TensorPtr(CVI_TENSOR* tensor) { return tensor->sys_mem; }
size_t CVI_NN_TensorSize(CVI_TENSOR* tensor) { return tensor->mem_size; }
size_t CVI_NN_TensorCount(CVI_TENSOR* tensor) { return tensor->count; }
CVI_SHAPE CVI_NN_TensorShape(CVI_TENSOR* tensor) { return tensor->shape; }
}

// ---- tests ----

static int failures = 0;
static void check(bool cond, const char* label) {
  printf("[%s] %s\n", cond ? "OK" : "FAIL", label);
  if (!cond) failures++;
}

static std::vector<uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

template <typename A, typename B>
static double snr_db(const A& ref, const B& test, size_t n) {
  double sig = 0, noise = 0;
  for (size_t i = 0; i < n; i++) {
    sig += (double)ref[i] * ref[i];
    noise += ((double)ref[i] - test[i]) * ((double)ref[i] - test[i]);
  }
  return 10 * std::log10(sig / (noise + 1e-30));
}

// [1, C, T] test vector -> [T, C] Mat.
static Mat channelFirstToTC(const WeightStore::Entry& e) {
  int C = e.shape[1], T = e.shape[2];
  Mat m(T, C);
  for (int c = 0; c < C; c++)
    for (int t = 0; t < T; t++) m.at(t, c) = e.at(c * T + t);
  return m;
}

int main() {
  std::vector<uint8_t> weights_buf = read_file("../research/weights.bin");
  std::vector<uint8_t> vectors_buf = read_file("../research/test_vectors.bin");
  std::vector<uint8_t> cmudict_buf = read_file("../research/cmudict.bin");
  WeightStore weights, vectors;
  check(weights.begin(weights_buf.data(), weights_buf.size()) && vectors.begin(vectors_buf.data(), vectors_buf.size()),
        "test data loaded (run from build_*/, like test_main)");
  Vocoder vocoder;
  vocoder.begin(weights);
  Flow flow;
  flow.begin(weights, /*n_flows=*/4, /*n_heads=*/2, /*window_size=*/4);
  g_vocoder = &vocoder;
  g_flow = &flow;
  Mat g = channelFirstToTC(*vectors.get("g"));

  printf("\n=== CviTpuVocoder ===\n");
  {
    CviTpuVocoder tpu;
    check(tpu.begin("fake:vocoder:64") && tpu.window() == 64, "CviTpuVocoder loads a [1,32,64,1] model");
    Mat z = channelFirstToTC(*vectors.get("z_ref"));
    auto ref = vocoder.forward(z, g);
    std::vector<float> out;
    bool ok = tpu.vocoderFn()(z, g, [&](const float* s, size_t n) { out.insert(out.end(), s, s + n); });
    double snr = out.size() == ref.size() ? snr_db(ref, out, ref.size()) : -1;
    printf("T=%d: SNR vs. CPU vocoder %.1f dB\n", z.rows(), snr);
    check(ok && snr > 100.0, "CviTpuVocoder matches the CPU vocoder");
    CviTpuVocoder bad;
    check(!bad.begin("fake:int8:64"), "CviTpuVocoder rejects a model with an INT8 output");
    check(!bad.begin("missing.cvimodel"), "CviTpuVocoder reports a model it can't load");
  }

  printf("\n=== CviTpuFlow ===\n");
  {
    CviTpuFlow tpu;
    check(tpu.add("fake:flow:256") && tpu.add("fake:flow:128") && tpu.maxLength() == 256, "CviTpuFlow loads two buckets");
    Mat z_p = channelFirstToTC(*vectors.get("z_p"));
    Mat ref = flow.reverse(z_p, g);
    Mat z;
    bool ok = tpu.flowFn()(z_p, g, z);
    double snr = ok && z.rows() == ref.rows() ? snr_db(ref.data(), z.data(), ref.data().size()) : -1;
    printf("T=%d: SNR vs. CPU flow %.1f dB\n", z_p.rows(), snr);
    check(ok && snr > 100.0, "CviTpuFlow matches the CPU flow");
    Mat too_long(300, z_p.cols());
    check(!tpu.flowFn()(too_long, g, z), "CviTpuFlow declines an utterance longer than its largest bucket");
  }

  printf("\n=== checkModel(): TPU models from another checkpoint ===\n");
  {
    TinyTTSCore core;  // the released model: 44.1 kHz, 512 samples per frame
    core.begin(weights_buf.data(), weights_buf.size(), cmudict_buf.data(), cmudict_buf.size());
    CviTpuVocoder same, named, other_hop, other_rate;
    check(same.begin("fake:vocoder:64") && same.checkModel(core), "checkModel() accepts a matching decoder");
    check(named.begin("fake:vocoder:64:0:decoder_44100hz_w64_bf16.cvimodel") && named.checkModel(core),
          "checkModel() accepts a decoder named for the weights' sample rate");
    check(other_hop.begin("fake:vocoder:64:256") && !other_hop.checkModel(core),
          "checkModel() rejects a decoder with another hop (e.g. a 16 kHz model)");
    check(other_rate.begin("fake:vocoder:64:0:decoder_22050hz_w64_bf16.cvimodel") && !other_rate.checkModel(core),
          "checkModel() rejects a decoder named for another sample rate");
    CviTpuFlow flow_same, flow_other;
    check(flow_same.add("fake:flow:128:0:flow_44100hz_l128_bf16.cvimodel") && flow_same.checkModel(core),
          "checkModel() accepts flow models named for the weights' sample rate");
    check(flow_other.add("fake:flow:128:0:flow_44100hz_l128_bf16.cvimodel") &&
              flow_other.add("fake:flow:256:0:flow_16000hz_l256_bf16.cvimodel") && !flow_other.checkModel(core),
          "checkModel() rejects flow models named for another sample rate");
  }

  printf("\n=== TinyTTSCore with both on the (fake) TPU ===\n");
  {
    TinyTTSCore cpu, tpu_core;
    cpu.begin(weights_buf.data(), weights_buf.size(), cmudict_buf.data(), cmudict_buf.size());
    tpu_core.begin(weights_buf.data(), weights_buf.size(), cmudict_buf.data(), cmudict_buf.size());
    g_vocoder = &tpu_core.vocoder();
    g_flow = &tpu_core.flow();
    CviTpuVocoder tpu_vocoder;
    CviTpuFlow tpu_flow;
    tpu_vocoder.begin("fake:vocoder:128");
    tpu_flow.add("fake:flow:256");
    tpu_core.setVocoder(tpu_vocoder.vocoderFn());
    tpu_core.setFlow(tpu_flow.flowFn());
    std::vector<float> a, b;
    SynthesisInfo ia = cpu.synthesize("Hello world!", [&](const float* s, size_t n) { a.insert(a.end(), s, s + n); });
    SynthesisInfo ib = tpu_core.synthesize("Hello world!", [&](const float* s, size_t n) { b.insert(b.end(), s, s + n); });
    double snr = a.size() == b.size() ? snr_db(a, b, a.size()) : -1;
    printf("y_len %d/%d, %zu/%zu samples, SNR %.1f dB\n", ia.y_len, ib.y_len, a.size(), b.size(), snr);
    check(ia.y_len == ib.y_len && snr > 100.0, "speak() through both TPU wrappers matches the CPU pipeline");
  }

  printf("\n%d check(s) failed.\n", failures);
  return failures == 0 ? 0 : 1;
}
