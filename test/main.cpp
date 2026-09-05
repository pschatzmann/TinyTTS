// Test harness for the real src/ library (not a parallel prototype).
// Validates against reference tensors dumped from the PyTorch model by
// research/export_weights_and_vectors.py and research/export_cmudict.py.
//
// Exercises TinyTTSCore directly (the portable orchestration layer, with
// stubbed duration_predictor/decoder callbacks) AND the top-level TinyTTS
// facade in src/TinyTTS.h -- including its real TFLite Micro interpreters,
// arenas, and allocation/deleter logic, run against the actual shipped
// .tflite models (research/tflite_int8/.../*.tflite). On a non-ARDUINO
// build TinyTTS.h's `tflite::` types come from the real TFLite Micro
// source, fetched at CMake-configure time (see cmake/FetchTFLiteMicro.cmake
// -- not tflm_esp32's ESP32-only precompiled binary), and its PSRAM
// allocation goes through plain malloc/free instead of heap_caps_malloc --
// see TinyTTS.h's class doc and tinyttsAlloc().
//
// Build, via CMake (from the repo root; also supports -DTINYTTS_ASAN=ON;
// first configure needs network access to fetch TFLite Micro):
//   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON
//   cmake --build build && ./build/test/test_main
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "TinyTTS.h"
#include "TinyTTS/Data.h"
#include "TinyTTS/DictionaryModel.h"
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
        for (int c = 0; c < C; c++) m.at(t, c) = e.data[(size_t)c * T + t];
    return m;
}

static Mat channel_first_1_to_row(const WeightStore::Entry& e) {
    // e.shape == [1, C, 1] -> Mat[1, C]
    int C = e.shape[1];
    Mat m(1, C);
    for (int c = 0; c < C; c++) m.at(0, c) = e.data[c];
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
    for (int i = 0; i < T; i++) phone_ids[i] = (int)std::lround(phone_e->data[i]);
    const WeightStore::Entry* tone_e = ref.vectors.get("tone_ids");
    for (int i = 0; i < T; i++) tone_ids[i] = (int)std::lround(tone_e->data[i]);
    const WeightStore::Entry* lang_e = ref.vectors.get("language_ids");
    for (int i = 0; i < T; i++) language_ids[i] = (int)std::lround(lang_e->data[i]);

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

// Duration -> frame-level expansion (the glue between duration_predictor
// and flow), using the PhonemeEncoder output from testPhonemeEncoder().
static void testAlignment(ReferenceData& ref, const PhonemeEncoderOutput& enc_out) {
    int T = ref.vectors.get("phone_ids")->shape[1];

    printf("\n=== Alignment (duration -> frame expansion) ===\n");
    const WeightStore::Entry* logw_e = ref.vectors.get("logw_ref");
    std::vector<float> logw(logw_e->data.begin(), logw_e->data.end());
    std::vector<int> durations = alignment::durationsFromLogw(logw);

    const WeightStore::Entry* dur_ref_e = ref.vectors.get("durations_ref");
    bool durations_match = true;
    for (int i = 0; i < T; i++) {
        int expected = (int)std::lround(dur_ref_e->data[i]);
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
// flow -> decoder) through TinyTTSCore directly, with stubbed
// duration_predictor/decoder (those are independently validated in Python
// already -- validate_int8.py/validate_tflite.py). Checks shapes flow
// through every stage correctly, not stage-level numerics.
static void testTinyTTSCoreOrchestration(const std::vector<uint8_t>& weights_buf,
                                          const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== TinyTTSCore end-to-end orchestration ===\n");
    TinyTTSCore tts;
    bool began = tts.begin(weights_buf.data(), weights_buf.size(), cmudict_buf.data(), cmudict_buf.size());
    check(began, "TinyTTSCore.begin()");

    Mat received_z, received_g;
    tts.setDurationPredictor([](const Mat& x, const Mat& g_in) {
        (void)g_in;
        return std::vector<float>(x.rows(), std::log(3.0f));  // ~3 frames/phoneme
    });
    tts.setDecoder([&](const Mat& z, const Mat& g_in) {
        received_z = z;
        received_g = g_in;
        return std::vector<float>(z.rows() * 512, 0.0f);  // dummy PCM, shape-only check
    });
    check(tts.isReady(), "TinyTTSCore.isReady() after wiring callbacks");

    // synthesize() decodes in fixed-size chunks (default 96 frames, see
    // TinyTTSCore::setDecoderChunkFrames) -- the stub decoder above is
    // called once per chunk, so `received_z` ends up holding whichever
    // chunk was decoded last (zero-padded to exactly 96 rows if it's a
    // short final chunk).
    size_t total_audio_samples = 0;
    auto on_audio = [&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    };
    SynthesisInfo info = tts.synthesize("Hello world!", on_audio, /*speaker_id=*/0);
    printf("synthesize(\"Hello world!\"): y_len=%d, decoder received z rows=%d cols=%d, total audio samples=%zu\n",
           info.y_len, received_z.rows(), received_z.cols(), total_audio_samples);
    check(info.y_len > 0 && received_z.rows() == 96 && total_audio_samples > 0,
          "full pipeline orchestration produces consistent shapes");
}

// The actual public TinyTTS facade, driving its own real TFLite Micro
// interpreters (not stubs) against the shipped quantized models -- on a
// non-ARDUINO build these come from the fetched TFLM source (see
// cmake/FetchTFLiteMicro.cmake), not tflm_esp32's ESP32-only precompiled
// binary, so this is a genuine end-to-end run of the exact
// interpreter-build/arena/deleter logic TinyTTS.h uses on real hardware,
// not just its orchestration.
static void testTinyTTSFacadeRealModels(const std::vector<uint8_t>& weights_buf,
                                         const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== TinyTTS facade (real TFLite Micro interpreters) ===\n");
    std::vector<uint8_t> dp_model_buf =
        read_file("../research/tflite_int8/duration_predictor/duration_predictor_full_integer_quant_with_int16_act.tflite");
    std::vector<uint8_t> decoder_model_buf =
        read_file("../research/tflite_int8/decoder/decoder_full_integer_quant_with_int16_act.tflite");

    TinyTTS<> tinytts_obj;
    tinytts_obj.setWeights(weights_buf.data(), weights_buf.size());
    tinytts_obj.setDictionary(cmudict_buf.data(), cmudict_buf.size());
    tinytts_obj.setDurationPredictorModel(dp_model_buf.data(), dp_model_buf.size(), /*max_phonemes=*/32);
    tinytts_obj.setDecoderModel(decoder_model_buf.data(), decoder_model_buf.size(), /*chunk_frames=*/96);

    size_t total_audio_samples = 0;
    bool began = tinytts_obj.begin([&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    check(began, "TinyTTS.begin() (builds real TFLite Micro interpreters)");
    bool spoke = tinytts_obj.speak("Hello world!");
    check(spoke, "TinyTTS.speak() returns true after begin()");
    printf("TinyTTS.speak(\"Hello world!\"): total audio samples=%zu\n", total_audio_samples);
    check(total_audio_samples > 0, "TinyTTS facade produces audio");
    tinytts_obj.end();
    check(!tinytts_obj.speak("should fail after end()"), "TinyTTS.speak() fails after end()");
}

// duration_predictor's TFLite model has a fixed 32-phoneme input window, but
// TinyTTS::runDurationPredictor() slides overlapping windows across longer
// text and keeps only each window's real-context "trusted middle" (see that
// method's doc) instead of truncating -- this confirms that actually works
// end-to-end, not just that it compiles: every phoneme in a text long
// enough to need multiple windows should get a real duration, not just the
// first 32.
static void testDurationPredictorLongText(const std::vector<uint8_t>& weights_buf,
                                           const std::vector<uint8_t>& cmudict_buf) {
    printf("\n=== duration_predictor sliding window (text > 32 phonemes) ===\n");
    std::vector<uint8_t> dp_model_buf =
        read_file("../research/tflite_int8/duration_predictor/duration_predictor_full_integer_quant_with_int16_act.tflite");
    std::vector<uint8_t> decoder_model_buf =
        read_file("../research/tflite_int8/decoder/decoder_full_integer_quant_with_int16_act.tflite");

    TinyTTS<> tts;
    tts.setWeights(weights_buf.data(), weights_buf.size());
    tts.setDictionary(cmudict_buf.data(), cmudict_buf.size());
    tts.setDurationPredictorModel(dp_model_buf.data(), dp_model_buf.size(), /*max_phonemes=*/32);
    tts.setDecoderModel(decoder_model_buf.data(), decoder_model_buf.size(), /*chunk_frames=*/96);

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
    check(expected_phonemes > 32, "test sentence actually exceeds one duration_predictor window (32 phonemes)");

    size_t total_audio_samples = 0;
    SynthesisInfo info = tts.core().synthesize(long_text, [&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    printf("long text: %zu phonemes, y_len=%d, total audio samples=%zu\n", expected_phonemes, info.y_len,
           total_audio_samples);
    check(info.durations.size() == expected_phonemes,
          "duration_predictor covers every phoneme in text longer than one window, not just the first 32");
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
    TinyTTS<> sketch_tts;
    sketch_tts.setWeights(tts_model_data::default_weights, tts_model_data::default_weights_len);
    sketch_tts.setDictionary(tts_model_data::default_cmudict_slim, tts_model_data::default_cmudict_slim_len);
    sketch_tts.setDictionaryModel(tts_model_data::default_dictionary_model,
                                   tts_model_data::default_dictionary_model_len);
    sketch_tts.setDurationPredictorModel(tts_model_data::default_duration_predictor_model,
                                          tts_model_data::default_duration_predictor_model_len, /*max_phonemes=*/32);
    sketch_tts.setDecoderModel(tts_model_data::default_decoder_model, tts_model_data::default_decoder_model_len,
                                /*chunk_frames=*/96);

    size_t total_audio_samples = 0;
    bool began = sketch_tts.begin([&](const float* samples, size_t count) {
        (void)samples;
        total_audio_samples += count;
    });
    check(began,
          "sketch-equivalent TinyTTS.begin() "
          "(default_weights/default_cmudict_slim/default_dictionary_model/"
          "default_duration_predictor_model/default_decoder_model)");
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
    testAlignment(ref, enc_out);
    testCmuDict(ref.cmudict_buf);
    testDictionaryModel();
    testTinyTTSCoreOrchestration(ref.weights_buf, ref.cmudict_buf);
    testTinyTTSFacadeRealModels(ref.weights_buf, ref.cmudict_buf);
    testDurationPredictorLongText(ref.weights_buf, ref.cmudict_buf);
    testTinyTTSFacadeSketchData();

    printf("\n%d check(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
