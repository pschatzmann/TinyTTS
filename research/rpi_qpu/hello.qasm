# Hello QPU: out[i] = in[i] * 2 + 1, one 16-float vector per iteration.
# Uniforms: qpu id, vector count, input address, output address.
# Each QPU writes through its own VPM row (row = qpu id).

mov ra0, unif               # qpu id
mov ra1, unif               # vector count
mov ra2, unif               # input address
mov ra3, unif               # output address
mov rb2, 64                 # bytes per vector

shl r0, elem_num, 2         # per-lane input address
add ra2, ra2, r0
mov r1, vpm_setup(1, 1, h32(0))
add rb0, r1, ra0            # VPM write setup for this QPU's row
shl r2, ra0, 7
mov r1, vdw_setup_0(1, 16, dma_h32(0, 0))
add rb1, r1, r2             # VPM -> memory DMA setup for the same row

:loop
mov t0s, ra2
ldtmu0
fmul r0, r4, 2.0
fadd r0, r0, 1.0
mov vw_setup, rb0
mov vpm, r0
mov -, mutex
mov vw_setup, rb1
mov vw_addr, ra3
mov -, vw_wait
mov mutex, 0
add ra2, ra2, rb2
add ra3, ra3, rb2
sub.setf ra1, ra1, 1
brr.anynz -, :loop
nop
nop
nop

thrend
mov interrupt, 1
nop
