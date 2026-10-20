// Switches the Milk-V Duo's Ethernet PHY pads (TX+/TX-/RX+/RX-) from the analog
// Ethernet PHY to digital I/O, so that I2S2 can use them. See docs/milkv-duo.md.
//
// Selecting IIS2_* in the pinmux (cvi-pinmux) alone is not enough: the pads stay
// connected to the PHY until it is put into "GPIO from top" mode here. Register
// sequence from the CV181x pinmux spreadsheet, as worked out by Eric Brombaugh
// (emeb) in https://community.milkv.io/t/i2s-milk-v-duo-speaker-max98357a/668/46,
// with the 0x808 offset corrected (the original used 0x806).
//
// Needs root (/dev/mem). Build:
//   riscv64-unknown-linux-musl-gcc -O2 -static -o eth2i2s eth2i2s.c
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define EPHY_BASE 0x03009000u

static volatile uint32_t *regs;

static void update(unsigned off, uint32_t clear, uint32_t set, const char *what) {
  volatile uint32_t *r = regs + off / 4;
  uint32_t before = *r;
  *r = (before & ~clear) | set;
  printf("0x%08x: 0x%08x -> 0x%08x  %s\n", EPHY_BASE + off, before, *r, what);
}

int main(void) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open /dev/mem");
    return 1;
  }
  void *map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, EPHY_BASE);
  if (map == MAP_FAILED) {
    perror("mmap");
    close(fd);
    return 1;
  }
  regs = (volatile uint32_t *)map;

  update(0x804, 0, 0x1, "rg_ephy_apb_rw_sel=1 (APB access)");
  update(0x808, 0x1f, 0x01, "rg_ephy_pll_stable_cnt=1 (10us)");
  update(0x800, 0xffffffffu, 0x0905, "rg_ephy_dig_rst_n=1 (release reset)");
  usleep(10);
  update(0x07c, 0x1f << 8, 0x05 << 8, "page_sel_mode0=5");
  update(0x078, 0xfff, 0xf00, "pads: GPIO from top");
  update(0x074, 0, 0x606, "RX+/RX-: input and output enable");
  update(0x070, 0, 0x606, "TX+/TX-: input and output enable");

  munmap(map, 4096);
  close(fd);
  return 0;
}
