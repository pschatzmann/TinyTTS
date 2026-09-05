// Test harness for the real src/ library (not a parallel prototype).
// Validates against reference tensors dumped from the PyTorch model by
// research/export_weights_and_vectors.py and research/export_cmudict.py.
//
// Exercises TinyTTSCore directly (the portable orchestration layer) AND the
// top-level TinyTTS facade in src/TinyTTS.h. All four model stages
// (text_encoder, flow, duration_predictor, decoder) are hand-written C++ --
// no TFLite Micro or other inference-runtime dependency anywhere, so this
// is a plain host build (no CMake fetch step needed for an inference
// runtime).
//
// Build, via CMake (from the repo root; also supports -DTINYTTS_ASAN=ON):
//   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON
//   cmake --build build && ./build/test/test_main
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "TinyTTS.h"
#include "TinyTTS/Data.h"
#include "TinyTTS/Decoder.h"
#include "TinyTTS/DictionaryModel.h"
#include "TinyTTS/DurationPredictor.h"
#include "TinyTTS/TinyTTSCore.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"

using namespace tinytts;

// ---- shared helpers ----

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read((char*)buf.data(), size);
    return buf;
}

static Mat channel_first_to_TC(const WeightStore::Entry& e) {
    // e.shape == [1, C, T] -> Mat[T, C]
    int C = e.shape[1], T = e.shape[2];
    Mat m(T, C);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) m.at(t, c) = e.at((size_t)c * T + t);
    return m;
}

static Mat channel_first_1_to_row(const WeightStore::Entry& e) {
    // e.shape == [1, C, 1] -> Mat[1, C]
    int C = e.shape[1];
    Mat m(1, C);
    for (int c = 0; c < C; c++) m.at(0, c) = e.at(c);
    return m;
}

static double cos_sim(const Mat& a, const Mat& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.data().size(); i++) {
        dot += (double)a.data()[i] * b.data()[i];
        na += (double)a.data()[i] * a.data()[i];
        nb += (double)b.data()[i] * b.data()[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static double max_abs_diff(const Mat& a, const Mat& b) {
    double m = 0;
    for (size_t i = 0; i < a.data().size(); i++) m = std::max(m, (double)std::fabs(a.data()[i] - b.data()[i]));
    return m;
}

static int failures = 0;
static void check(bool cond, const char* label) {
    printf("[%s] %s\n", cond ? "OK" : "FAIL", label);
    if (!cond) failures++;
}

// ---- reference data shared across several test cases ----

struct ReferenceData {
    std::vector<uint8_t> weights_buf;
    std::vector<uint8_t> vectors_buf;
    std::vector<uint8_t> cmudict_buf;
    WeightStore weights, vectors;
};

static ReferenceData loadReferenceData() {
    ReferenceData r;
    r.weights_buf = read_file("../research/weights.bin");
    r.vectors_buf = read_file("../research/test_vectors.bin");
    r.cmudict_buf = read_file("../research/cmudict.bin");
    r.weights.begin(r.weights_buf.data(), r.weights_buf.size());
    r.vectors.begin(r.vectors_buf.data(), r.vectors_buf.size());
    return r;
}

// ---- test cases ----

// PhonemeEncoder forward pass against the PyTorch reference. Returns the
// output so testAlignment() (which needs enc_out.m_p/logs_p) can reuse it
// without recomputing.
static PhonemeEncoderOutput testPhonemeEncoder(ReferenceData& ref, int n_heads, int window_size) {
    PhonemeEncoder enc_w;
    enc_w.begin(ref.weights, n_heads, window_size);

    const WeightStore::Entry* phone_e = ref.vectors.get("phone_ids");
    int T = phone_e->shape[1];
    std::vector<int> phone_ids(T), tone_ids(T), language_ids(T);
    for (int i = 0; i < T; i++) phone_ids[i] = (int)std::lround(phone_e->at(i));
    const WeightStore::Entry* tone_e = ref.vectors.get("tone_ids");
    for (int i = 0; i < T; i++) tone_ids[i] = (int)std::lround(tone_e->at(i));
    const WeightStore::Entry* lang_e = ref.vectors.get("language_ids");
    for (int i = 0; i < T; i++) language_ids[i] = (int)std::lround(lang_e->at(i));

    Mat g = channel_first_1_to_row(*ref.vectors.get("g"));

    printf("=== PhonemeEncoder (T=%d) ===\n", T);
    PhonemeEncoderOutput enc_out = enc_w.forward(phone_ids, tone_ids, language_ids, g);

    Mat x_ref = channel_first_to_TC(*ref.vectors.get("x_ref"));
    Mat m_p_ref = channel_first_to_TC(*ref.vectors.get("m_p_ref"));
    Mat logs_p_ref = channel_first_to_TC(*ref.vectors.get("logs_p_ref"));

    printf("cos_sim x:      %.6f\n", cos_sim(enc_out.x, x_ref));
    printf("cos_sim m_p:    %.6f  max_abs_diff=%.6f\n", cos_sim(enc_out.m_p, m_p_ref), max_abs_diff(enc_out.m_p, m_p_ref));
    printf("cos_sim logs_p: %.6f  max_abs_diff=%.6f\n", cos_sim(enc_out.logs_p, logs_p_ref),
           max_abs_diff(enc_out.logs_p, logs_p_ref));
    check(cos_sim(enc_out.x, x_ref) > 0.9999, "PhonemeEncoder.x matches reference");
    check(cos_sim(enc_out.m_p, m_p_ref) > 0.9999, "PhonemeEncoder.m_p matches reference");
    check(cos_sim(enc_out.logs_p, logs_p_ref) > 0.9999, "PhonemeEncoder.logs_p matches reference");
    return enc_out;
}

static void testFlow(ReferenceData& ref, int n_flows, int n_heads, int window_size) {
    Flow flow_w;
    flow_w.begin(ref.weights, n_flows, n_heads, window_size);

    Mat g = channel_first_1_to_row(*ref.vectors.get("g"));
    printf("\n=== Flow (reverse) ===\n");
    Mat z_p = channel_first_to_TC(*ref.vectors.get("z_p"));
    Mat z = flow_w.reverse(z_p, g);
    Mat z_ref = channel_first_to_TC(*ref.vectors.get("z_ref"));
    printf("cos_sim z:      %.6f  max_abs_diff=%.6f\n", cos_sim(z, z_ref), max_abs_diff(z, z_ref));
    check(cos_sim(z, z_ref) > 0.9999, "Flow.reverse matches reference");
}

// Hand-written Decoder forward pass against the PyTorch reference
// (tiny_tts.models.synthesizer.WaveformDecoder, `net_g.dec`). Replaces the
// TFLite Micro decoder model -- plain Conv1d/ConvTranspose1d/LeakyReLU, no
// TFLM fixed-window constraint, run in one pass over the whole utterance.
// Uses the reference flow output (z_ref) as input, so this test is isolated
// to decoder correctness regardless of Flow's own (separately validated)
// output.
static void testDecoder(ReferenceData& ref) {
    Decoder dec;
    dec.begin(ref.weights);

    Mat g = channel_first_1_to_row(*ref.vectors.get("g"));
    Mat z_ref = channel_first_to_TC(*ref.vectors.get("z_ref"));
    printf("\n=== Decoder ===\n");
    auto audio = dec.forward(z_ref, g);

    const WeightStore::Entry* audio_ref_e = ref.vectors.get("audio_ref");
    int n = (int)audio.size();
    double dot = 0, na = 0, nb = 0, max_diff = 0;
    for (int i = 0; i < n; i++) {
        double a = audio[i], b = audio_ref_e->at(i);
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_diff = std::max(max_diff, std::fabs(a - b));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    printf("cos_sim audio:  %.6f  max_abs_diff=%.6f  n_samples=%d\n", cos, max_diff, n);
    check((int)audio_ref_e->count == n, "Decoder.forward output length matches reference");
    check(cos > 0.999, "Decoder.forward matches reference");
}

// Hand-written DurationPredictor forward pass against the PyTorch reference
// (tiny_tts.models.synthesizer.DurationEstimator, `net_g.dp`). Replaces the
// TFLite Micro duration_predictor model -- plain Conv1d/ChannelNorm/ReLU, no
// TFLM fixed-window constraint.
static std::vector<float> testDurationPredictor(ReferenceData& ref, const PhonemeEncoderOutput& enc_out) {
    DurationPredictor dp;
    dp.begin(ref.weights);

    Mat g = channel_first_1_to_row(*ref.vectors.get("g"));
    printf("\n=== DurationPredictor ===\n");
    std::vector<float> logw = dp.forward(enc_out.x, g);

    const WeightStore::Entry* logw_ref_e = ref.vectors.get("logw_ref");
    int T = (int)logw.size();
    double dot = 0, na = 0, nb = 0, max_diff = 0;
    for (int i = 0; i < T; i++) {
        double a = logw[i], b = logw_ref_e->at(i);
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_diff = std::max(max_diff, std::fabs(a - b));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    printf("cos_sim logw:   %.6f  max_abs_diff=%.6f\n", cos, max_diff);
    check(cos > 0.9999, "DurationPredictor.forward matches reference");
    return logw;
}

// Duration -> frame-level expansion (the glue between duration_predictor
// and flow), using the PhonemeEncoder output from testPhonemeEncoder() and
// the hand-written DurationPredictor's own logw (not the reference logw --
// this exercises the real end-to-end hand-written path).
static void testAlignment(ReferenceData& ref, const PhonemeEncoderOutput& enc_out, const std::vector<float>& logw) {
    int T = ref.vectors.get("phone_ids")->shape[1];

    printf("\n=== Alignment (duration -> frame expansion) ===\n");
    std::vector<int> durations = alignment::durationsFromLogw(logw);

    const WeightStore::Entry* dur_ref_e = ref.vectors.get("durations_ref");
    bool durations_match = true;
    for (int i = 0; i < T; i++) {
        int expected = (int)std::lround(dur_ref_e->at(i));
        if (durations[i] != expected) durations_match = false;
    }
    check(durations_match, "durations match reference");

    int T_y = alignment::totalDuration(durations);
    printf("computed T_y=%d\n", T_y);

    Mat m_p_exp = alignment::expandByDuration(enc_out.m_p, durations, T_y);
    Mat logs_p_exp = alignment::expandByDuration(enc_out.logs_p, durations, T_y);
    Mat m_p_exp_ref = channel_first_to_TC(*ref.vectors.get("m_p_exp_ref"));
    Mat logs_p_exp_ref = channel_first_to_TC(*ref.vectors.get("logs_p_exp_ref"));

    printf("cos_sim m_p_exp:    %.6f  max_abs_diff=%.6f\n", cos_sim(m_p_exp, m_p_exp_ref),
           max_abs_diff(m_p_exp, m_p_exp_ref));
    printf("cos_sim logs_p_exp: %.6f  max_abs_diff=%.6f\n", cos_sim(logs_p_exp, logs_p_exp_ref),
           max_abs_diff(logs_p_exp, logs_p_exp_ref));
    check(cos_sim(m_p_exp, m_p_exp_ref) > 0.9999, "alignment m_p_exp matches reference");
    check(cos_sim(logs_p_exp, logs_p_exp_ref) > 0.9999, "alignment logs_p_exp matches reference");
}

static void testCmuDict(const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== CmuDict lookup ===\n");
    CmuDict dict;
    check(dict.begin(cmudict_buf.data(), cmudict_buf.size()), "CmuDict.begin() parses cmudict.bin");
    printf("word count: %d\n", dict.wordCount());
    CmuDict::Entry entry;
    bool found_hello = dict.lookup("HELLO", &entry);
    check(found_hello, "CmuDict finds HELLO");
    if (found_hello) {
        printf("HELLO -> %d phones: ", entry.count);
        for (int i = 0; i < entry.count; i++)
            printf("(sym=%d,tone=%d) ", entry.symbol_tone_pairs[i * 2], entry.symbol_tone_pairs[i * 2 + 1]);
        printf("\n");
    }
    check(!dict.lookup("ZZZZNOTAWORDZZZZ", &entry), "CmuDict correctly rejects a nonsense word");
}

// Neural G2P fallback (for words not in CmuDict) against reference
// (symbol_id, tone) sequences computed by research/g2p_reference_numpy.py
// -- itself verified bit-exact against the actual Python `g2p_en` package's
// output before being trusted (see docs/research.md), the same "verify
// against ground truth before porting" approach as every other model here.
static void testDictionaryModel() {
    printf("\n=== DictionaryModel (neural G2P fallback) ===\n");
    std::vector<uint8_t> dict_model_buf = read_file("../research/dictionary_model.bin");

    DictionaryModel model;
    check(model.begin(dict_model_buf.data(), dict_model_buf.size()), "DictionaryModel.begin()");

    struct Case {
        const char* word;
        std::vector<std::pair<int, int>> expected;  // (symbol_id, tone)
    };
    std::vector<Case> cases = {
        {"arduino", {{21, 1}, {85, 0}, {34, 0}, {103, 1}, {65, 2}, {73, 0}, {80, 1}}},
        {"esp32", {{39, 2}, {87, 0}, {82, 0}}},
        {"tinytts", {{89, 0}, {29, 2}, {73, 0}, {59, 1}, {89, 0}, {87, 0}}},
        {"zzznotaword",
         {{111, 0}, {23, 1}, {73, 0}, {89, 0}, {21, 2}, {46, 0}, {108, 0}, {21, 3}, {71, 0}, {43, 1}, {34, 0}}},
        {"blorpington",
         {{30, 0}, {85, 0}, {27, 2}, {70, 0}, {82, 0}, {85, 0}, {59, 1}, {74, 0}, {89, 0}, {23, 1}, {73, 0}}},
    };
    for (const auto& c : cases) {
        auto predicted = model.predict(c.word);
        bool match = predicted.size() == c.expected.size();
        if (match) {
            for (size_t i = 0; i < predicted.size(); i++) {
                if (predicted[i].symbol_id != c.expected[i].first || predicted[i].tone != c.expected[i].second) {
                    match = false;
                    break;
                }
            }
        }
        printf("%s -> ", c.word);
        for (auto& p : predicted) printf("(sym=%d,tone=%d) ", p.symbol_id, p.tone);
        printf("\n");
        check(match, (std::string("DictionaryModel.predict(\"") + c.word + "\") matches reference").c_str());
    }
}

// Full orchestration (G2P -> encoder -> duration_predictor -> alignment ->
// flow -> decoder) through TinyTTSCore directly -- every stage is the real
// hand-written model, loaded from weights_buf by begin() (each independently
// validated against the PyTorch reference above: testDurationPredictor(),
// testDecoder(), etc.). Checks shapes/output are consistent end-to-end, not
// stage-level numerics again.
static void testTinyTTSCoreOrchestration(const std::vector<uint8_t>& weights_buf,
                                          const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== TinyTTSCore end-to-end orchestration ===\n");
    TinyTTSCore tts;
    bool began = tts.begin(weights_buf.data(), weights_buf.size(), cmudict_buf.data(), cmudict_buf.size());
    check(began, "TinyTTSCore.begin()");
    check(tts.isReady(), "TinyTTSCore.isReady() after begin()");

    size_t total_audio_samples = 0;
    auto on_audio = [&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    };
    SynthesisInfo info = tts.synthesize("Hello world!", on_audio, /*speaker_id=*/0);
    printf("synthesize(\"Hello world!\"): y_len=%d, total audio samples=%zu\n", info.y_len, total_audio_samples);
    check(info.y_len > 0 && total_audio_samples > 0, "full pipeline orchestration produces consistent output");
}

// The actual public TinyTTS facade (not TinyTTSCore directly) -- checks the
// Print-free portable begin()/speak()/end() lifecycle on top of the real
// hand-written models, since testTinyTTSCoreOrchestration() above already
// covers the orchestration itself.
static void testTinyTTSFacadeRealModels(const std::vector<uint8_t>& weights_buf,
                                         const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== TinyTTS facade ===\n");
    TinyTTS tinytts_obj;
    tinytts_obj.setWeights(weights_buf.data(), weights_buf.size());
    tinytts_obj.setDictionary(cmudict_buf.data(), cmudict_buf.size());

    size_t total_audio_samples = 0;
    bool began = tinytts_obj.begin([&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    check(began, "TinyTTS.begin()");
    bool spoke = tinytts_obj.speak("Hello world!");
    check(spoke, "TinyTTS.speak() returns true after begin()");
    printf("TinyTTS.speak(\"Hello world!\"): total audio samples=%zu\n", total_audio_samples);
    check(total_audio_samples > 0, "TinyTTS facade produces audio");
    tinytts_obj.end();
    check(!tinytts_obj.speak("should fail after end()"), "TinyTTS.speak() fails after end()");
}

// Every stage (duration_predictor and decoder included) is hand-written C++,
// run in a single pass over the whole utterance -- no TFLite Micro
// fixed-input-shape window to hit anywhere in the pipeline. This test uses a
// long sentence (comfortably past the 32-phoneme window the old TFLM
// duration_predictor used to have) to confirm there's no length limit left
// to trip.
static void testLongText(const std::vector<uint8_t>& weights_buf, const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== Long text, single pass ===\n");

    TinyTTS tts;
    tts.setWeights(weights_buf.data(), weights_buf.size());
    tts.setDictionary(cmudict_buf.data(), cmudict_buf.size());

    bool began = tts.begin([](const float*, size_t) {});
    check(began, "TinyTTS.begin() for long-text test");

    const std::string long_text =
        "The quick brown fox jumps over the lazy dog and then runs away quickly into the "
        "dark forest before anyone notices what has happened";

    // Expected phoneme count -- the same computation synthesize() does
    // internally, via the same public G2P pipeline -- so this doesn't just
    // check "no crash", it confirms every phoneme gets covered.
    G2POutput g2p_out = tts.core().g2p().process(long_text);
    size_t expected_phonemes = TextG2P::insertBlanks(g2p_out.phone_ids).size();
    check(expected_phonemes > 32, "test sentence exceeds the old TFLM duration_predictor window size (32 phonemes)");

    size_t total_audio_samples = 0;
    SynthesisInfo info = tts.core().synthesize(long_text, [&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    printf("long text: %zu phonemes, y_len=%d, total audio samples=%zu\n", expected_phonemes, info.y_len,
           total_audio_samples);
    check(info.durations.size() == expected_phonemes,
          "every phoneme in text longer than the old window size gets covered");
    check(total_audio_samples > 0, "long text still produces audio");
}

// Mirrors examples/tts_i2s_output/tts_i2s_output.ino's setup(): same setter
// calls, in the same order, against the exact generated headers the shipped
// sketch includes (the slimmed dictionary + neural G2P model combo, not
// TinyTTS/Data.h's full-dictionary umbrella -- see that sketch's own
// comment for why) -- not research/'s raw .bin/.tflite files like
// testTinyTTSFacadeRealModels() above, so a bug in export_headers.py itself
// (wrong source file, broken encoding, ...) would be caught here even
// though that section wouldn't catch it. begin(AudioChunkFn) stands in for
// the sketch's begin(Print&) (Arduino/I2S-only, no host equivalent) --
// begin(Print&) is a thin wrapper around this same portable overload (see
// TinyTTS.h), so this still exercises the exact code path.
static void testTinyTTSFacadeSketchData() {
    printf("\n=== Sketch-equivalent facade (examples/tts_i2s_output data) ===\n");
    TinyTTS sketch_tts;
    sketch_tts.setWeights(default_weights, default_weights_len);
    sketch_tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
    sketch_tts.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);

    size_t total_audio_samples = 0;
    bool began = sketch_tts.begin([&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    check(began,
          "sketch-equivalent TinyTTS.begin() "
          "(default_weights/default_cmudict_slim/default_dictionary_model)");
    bool spoke = sketch_tts.speak("Hello world!");
    check(spoke, "sketch-equivalent TinyTTS.speak(\"Hello world!\")");
    printf("total audio samples=%zu\n", total_audio_samples);
    check(total_audio_samples > 0, "sketch-equivalent facade produces audio");
}

int main() {
    const int n_heads = 2, window_size = 4, n_flows = 4;

    ReferenceData ref = loadReferenceData();
    PhonemeEncoderOutput enc_out = testPhonemeEncoder(ref, n_heads, window_size);
    testFlow(ref, n_flows, n_heads, window_size);
    testDecoder(ref);
    std::vector<float> logw = testDurationPredictor(ref, enc_out);
    testAlignment(ref, enc_out, logw);
    testCmuDict(ref.cmudict_buf);
    testDictionaryModel();
    testTinyTTSCoreOrchestration(ref.weights_buf, ref.cmudict_buf);
    testTinyTTSFacadeRealModels(ref.weights_buf, ref.cmudict_buf);
    testLongText(ref.weights_buf, ref.cmudict_buf);
    testTinyTTSFacadeSketchData();

    printf("\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
