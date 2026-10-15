#pragma once
// Linux on a Raspberry Pi 1 / Zero (BCM2835, VideoCore IV) only --
// deliberately not included by TinyTTS.h. Needs root (/dev/vcio, /dev/mem)
// and the vc4-kms-v3d overlay disabled, see docs/rpi-zero-gpu.md.
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace tinytts {

/**
 * @brief One block of GPU memory, mapped into this process, and running QPU
 * programs on it.
 *
 * The block is allocated with the firmware's allocator and mapped through
 * /dev/mem uncached, so ARM writes are visible to the QPUs without a flush.
 * Addresses handed to the QPUs are bus addresses: busAddress(ptr).
 *
 * Memory and QPU power go through the firmware mailbox (/dev/vcio, as in
 * hello_fft's mailbox.c), the programs through the V3D registers.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class QpuDevice {
 public:
  /// MEM_FLAG_L1_NONALLOCATING (0xC): the QPUs see the block through the
  /// L2-cached alias, as hello_fft does on the BCM2835.
  static constexpr uint32_t kMemFlags = 0xC;

  ~QpuDevice() { end(); }

  bool begin(uint32_t size) {
    mb_ = open("/dev/vcio", O_RDWR);
    if (mb_ < 0) return fail("can't open /dev/vcio (run as root)");
    if (property(0x30012, 1) != 0) return fail("qpu_enable failed");
    enabled_ = true;
    size_ = (size + 4095) & ~4095u;
    handle_ = property(0x3000c, size_, 4096, kMemFlags);
    if (!handle_ || handle_ == ~0u) return fail("mem_alloc failed (gpu_mem too small?)");
    bus_ = property(0x3000d, handle_);
    if (!bus_ || bus_ == ~0u) return fail("mem_lock failed");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return fail("can't open /dev/mem");
    void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, bus_ & ~0xC0000000u);
    close(fd);
    if (p == MAP_FAILED) return fail("mmap failed");
    arm_ = (uint8_t*)p;
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return fail("can't open /dev/mem");
    p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, kV3dBase);
    close(fd);
    if (p == MAP_FAILED) return fail("mmap of the V3D registers failed");
    v3d_ = (volatile uint32_t*)p;
    if (v3d_[kIdent0] != 0x02443356) return fail("unexpected V3D_IDENT0 0x%08x", v3d_[kIdent0]);
    if (v3d_[kSrqcs] & 0x3f) return fail("QPU programs still queued from an earlier run (V3D_SRQCS 0x%08x) -- reboot", v3d_[kSrqcs]);
    // The firmware reserves all QPUs against user programs (bit 0 of each
    // QPU's 4-bit field); lift that until end().
    sqrsv_[0] = v3d_[kSqrsv0];
    sqrsv_[1] = v3d_[kSqrsv1];
    v3d_[kSqrsv0] = sqrsv_[0] & ~0x11111111u;
    v3d_[kSqrsv1] = sqrsv_[1] & ~0x11111111u;
    reserved_ = true;
    // No host interrupts from the programs' final `mov interrupt, 1`.
    v3d_[kDbcfg] = 0;
    v3d_[kDbqite] = 0;
    v3d_[kDbqitc] = ~0u;
    return true;
  }

  void end() {
    if (reserved_) {
      v3d_[kSqrsv0] = sqrsv_[0];
      v3d_[kSqrsv1] = sqrsv_[1];
    }
    reserved_ = false;
    if (v3d_) munmap((void*)v3d_, 4096);
    v3d_ = nullptr;
    if (arm_) munmap(arm_, size_);
    arm_ = nullptr;
    if (bus_) property(0x3000e, handle_);
    bus_ = 0;
    if (handle_) property(0x3000f, handle_);
    handle_ = 0;
    if (enabled_) property(0x30012, 0);
    enabled_ = false;
    if (mb_ >= 0) close(mb_);
    mb_ = -1;
  }

  uint8_t* data() { return arm_; }
  uint32_t size() const { return size_; }

  /// Bus address of `p`, which must point into data().
  uint32_t busAddress(const void* p) const { return bus_ + (uint32_t)((const uint8_t*)p - arm_); }

  /// Runs `num_qpus` programs (at most 16) and waits until all have ended.
  /// `mail` holds `num_qpus` pairs of (uniforms bus address, code bus
  /// address).
  ///
  /// Through the V3D user program queue, not the firmware's execute_qpu:
  /// on current firmware (2026-05) that returns before the programs have
  /// finished.
  bool execute(uint32_t num_qpus, const uint32_t* mail, uint32_t timeout_ms = 10000) {
    v3d_[kL2cactl] = 1 << 2;          // clear the V3D L2 cache: the ARM wrote new data
    v3d_[kSlcactl] = 0x0f0f0f0f;      // and the per-slice TMU, uniform and instruction caches
    v3d_[kSrqcs] = (1 << 7) | (1 << 8) | (1 << 16);  // reset error, request and completion counts
    for (uint32_t q = 0; q < num_qpus; q++) {
      v3d_[kSrqua] = mail[2 * q];
      v3d_[kSrqul] = 1024;
      v3d_[kSrqpc] = mail[2 * q + 1];  // queues the program
    }
    auto start = std::chrono::steady_clock::now();
    while (((v3d_[kSrqcs] >> 16) & 0xff) != num_qpus) {
      if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms))
        return fail("QPU programs timed out (V3D_SRQCS 0x%08x)", v3d_[kSrqcs]);
    }
    if (v3d_[kSrqcs] & (1 << 7)) return fail("QPU program queue error (V3D_SRQCS 0x%08x)", v3d_[kSrqcs]);
    return true;
  }

  /// The firmware's execute_qpu, for comparison: returns early, see execute().
  bool executeMailbox(uint32_t num_qpus, const uint32_t* mail, uint32_t timeout_ms = 10000) {
    uint32_t r = property(0x30011, num_qpus, busAddress(mail), /*noflush=*/1, timeout_ms);
    if (r != 0) return fail("execute_qpu failed or timed out (0x%08x)", r);
    return true;
  }

 private:
  static constexpr unsigned long kMboxProperty = 0xc0046400;  // _IOWR(100, 0, char*) on 32-bit
  static constexpr off_t kV3dBase = 0x20c00000;  // BCM2835 ARM physical address
  // V3D register indices (byte offset / 4), VideoCore IV 3D Architecture Reference Guide.
  static constexpr int kIdent0 = 0x000 / 4, kL2cactl = 0x020 / 4, kSlcactl = 0x024 / 4;
  static constexpr int kSqrsv0 = 0x410 / 4, kSqrsv1 = 0x414 / 4;
  static constexpr int kSrqpc = 0x430 / 4, kSrqua = 0x434 / 4, kSrqul = 0x438 / 4, kSrqcs = 0x43c / 4;
  static constexpr int kDbcfg = 0xe00 / 4, kDbqite = 0xe2c / 4, kDbqitc = 0xe30 / 4;
  int mb_ = -1;
  bool enabled_ = false;
  uint32_t size_ = 0, handle_ = 0, bus_ = 0;
  uint8_t* arm_ = nullptr;
  volatile uint32_t* v3d_ = nullptr;
  uint32_t sqrsv_[2] = {};
  bool reserved_ = false;

  /// One mailbox property tag with up to 4 values; returns the first value
  /// of the response.
  uint32_t property(uint32_t tag, uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0) {
    uint32_t n = tag == 0x3000c ? 3 : tag == 0x30011 ? 4 : 1;
    uint32_t msg[10] = {(6 + n) * 4, 0, tag, n * 4, n * 4, a, b, c, d, 0};
    msg[5 + n] = 0;  // end tag
    if (ioctl(mb_, kMboxProperty, msg) < 0) return ~0u;
    return msg[5];
  }

  bool fail(const char* fmt, uint32_t v = 0) {
    std::fprintf(stderr, "QpuDevice: ");
    std::fprintf(stderr, fmt, v);
    std::fprintf(stderr, "\n");
    return false;
  }
};

}  // namespace tinytts
