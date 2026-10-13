#!/usr/bin/env python3
"""Generates the QPU kernels of the Raspberry Pi Zero vocoder (QpuVocoder).

Writes qpu_kernels.qasm (assemble with vc4asm, see CMakeLists.txt) and
qpu_kernels_index.h (entry point of each variant, in instructions).

One kernel family computes conv1d and ConvTranspose1d layers. Activations are
channel-major float32 rows with kMargin (32) zeros on both sides:
  value (c, t) at buf[c * row + 32 + t], row = T16 + 64, T16 = T rounded up to 16.

Each QPU runs a list of jobs. A job is a range of blocks of one channel tile:
16 consecutive input times m (one per SIMD lane) and N accumulators
(ra0..ra(N-1)). Per tap the 16 inputs come from the TMU (prefetched two taps
ahead) and the N weights from the uniform stream, broadcast to all lanes:
one multiply-add per instruction (fmul on the MUL ALU, the previous product
added on the ADD ALU). The weight stream, rewound with unif_addr for every
block, is
  off(0), off(1), bias[N], then per tap i: off(i+2), w[0..N-1][tap i]
where off(i) is tap i's byte offset from the block's input address
(x + (32 + m) * 4, plus 4 * lane). The host builds it (QpuConv.h):
  conv1d:          accumulator a = output channel, taps (ci, kk)
  ConvTranspose1d: accumulator a = col * s + r for output time m * s + r,
                   taps (ci, d) reading input m + d (polyphase)

After the taps, the epilogue (variant flags) computes per accumulator
  v = acc (+ R if RES); if SUM: v = v * scale (+ S if SUMACC)
zeroes lanes past the end of the sequence (so the zero margins of the next
layer's input stay zero), and stores v (OUT0) and/or leaky_relu(v, alpha)
(OUT1). Stores go through the VPM (shared, under the hardware mutex) and the
VPM->memory DMA: one row per channel, or for ConvTranspose1d one vertical
DMA per channel that interleaves the s phases.

Job (16 words, see QpuConv.h): x, out0, out1, R, S (addresses for the job's
first block), block count, weight stream, tap loop count, row stride (bytes),
remaining valid input times, alpha, scale. With TM (time-major output, for
the flow's keys and values) the row stride slot holds the DMA stride setup
word and the scale slot the output advance per block.
QPU uniforms: job list address, job count.

Two more kernels serve the flow (QpuFlow.h): attention() and layernorm(),
see there.

Usage: gen_qpu_kernels.py <out_dir>
"""
import os
import re
import sys

RES, SUM, SUMACC, OUT0, OUT1, TM = 1, 2, 4, 8, 16, 32
FLAG_NAMES = {RES: "RES", SUM: "SUM", SUMACC: "SUMACC", OUT0: "OUT0", OUT1: "OUT1", TM: "TM"}
KIND_CONV, KIND_ATTENTION, KIND_LAYERNORM = 0, 1, 2

# Epilogues the vocoder (QpuVocoder.h) and the flow (QpuFlow.h) use.
CONV_FLAGS = [
    RES | OUT0,                     # flow: attention output, FFN conv_2, post
    OUT0,                           # conv_post
    OUT1,                           # conv_pre, resblock convs1
    RES | OUT0 | OUT1,              # resblock convs2 (pairs 0, 1)
    RES | SUM | OUT0,               # last convs2 of resblock 0: S = x / 3
    RES | SUM | SUMACC | OUT0,      # ... of resblock 1: S += x / 3
    RES | SUM | SUMACC | OUT1,      # ... of resblock 2: leaky(S + x / 3)
]
VARIANTS = ([(n, 1, f) for n in (16, 8, 4, 2) for f in CONV_FLAGS] +
            [(n, s, OUT0 | OUT1) for s, ns in ((2, (16, 8, 4)), (4, (16, 8)), (8, (16,))) for n in ns] +
            [(16, 1, OUT0 | TM)])  # flow: keys and values, time-major


def unroll_for(n):
    return max(1, 16 // (n + 2))


def name(n, s, flags):
    return f"n{n}_s{s}_f{flags}"


def tap_body(n, first):
    """One tap. r1 holds the previous tap's last product."""
    ins = [f"fadd ra{n-1}, ra{n-1}, r1; ldtmu0",  # previous tap's last add; x -> r4
           "add t0s, rb0, unif"]                 # prefetch tap i+2
    for j in range(n):
        add = f"fadd ra{j-1}, ra{j-1}, r1" if j > 0 else ("sub.setf r3, r3, 1" if first else "nop")
        ins.append(f"{add}; fmul r1, r4, unif")
    return ins


def epilogue(n, s, flags):
    out = []
    groups = [list(range(i, min(i + 4, n))) for i in range(0, n, 4)]  # TMU FIFO: 4 requests
    if flags & (RES | SUMACC):
        if flags & RES:
            out.append("mov r0, rb3")
        if flags & SUMACC:
            out.append("mov r2, rb4")
        for g in groups:
            for _ in g:
                if flags & RES:
                    out.append("add r0, r0, rb7; mov t0s, r0")
                if flags & SUMACC:
                    out.append("add r2, r2, rb7; mov t1s, r2")
            if flags & RES:
                for a in g:
                    out += ["ldtmu0", f"fadd ra{a}, ra{a}, r4"]
            if flags & SUM:
                out += [f"fmul ra{a}, ra{a}, rb10" for a in g]
            if flags & SUMACC:
                for a in g:
                    out += ["ldtmu1", f"fadd ra{a}, ra{a}, r4"]
    elif flags & SUM:
        out += [f"fmul ra{a}, ra{a}, rb10" for a in range(n)]
    elif flags & RES:
        raise ValueError("RES needs the TMU path")

    # Zero the lanes at or past the end of the sequence.
    out.append("sub.setf -, elem_num, rb8")
    out += [f"mov.ifnn ra{a}, 0" for a in range(n)]

    out += ["mov -, mutex", "mov vw_setup, vpm_setup(0, 1, h32(0))"]
    if flags & OUT0:
        out += [f"mov vpm, ra{a}" for a in range(n)]
    if flags & OUT1:
        for a in range(n):
            out += [f"fmul r3, ra{a}, rb9", f"fmax vpm, ra{a}, r3"]

    if flags & TM:
        # Time-major: 16 columns (times) of n channels each, one memory row
        # per time; rb7 holds the VDW stride setup word for the row gap.
        out += [f"mov vw_setup, vdw_setup_0(16, {n}, dma_v32(0, 0))",
                "mov vw_setup, rb7",
                "mov vw_addr, rb1",
                "mov -, vw_wait",
                "mov mutex, 0"]
        return out

    row = 0
    if s > 1:
        out.append("mov vw_setup, vdw_setup_1(0)")
    for flag, base in ((OUT0, "rb1"), (OUT1, "rb2")):
        if not flags & flag:
            continue
        out.append(f"mov r2, {base}")
        channels = n // s
        for c in range(channels):
            if s == 1:
                setup = f"vdw_setup_0(1, 16, dma_h32({row}, 0))"
            else:  # 16 columns of s phases each -> 16 * s consecutive samples
                setup = f"vdw_setup_0(16, {s}, dma_v32({row}, 0))"
            out += [f"mov vw_setup, {setup}", "mov vw_addr, r2", "mov -, vw_wait"]
            if c < channels - 1:
                out.append("add r2, r2, rb7")
            row += s
    out.append("mov mutex, 0")
    return out


def generate(n, s, flags):
    p = name(n, s, flags)
    u = unroll_for(n)
    body = []
    for k in range(u):
        body += tap_body(n, k == 0)
    loop = body[:-3] + [f"brr.anynz -, :{p}_tap"] + body[-3:]  # last 3 = delay slots
    flag_text = "|".join(v for k, v in FLAG_NAMES.items() if flags & k)
    out = [f"# {p}: {n} accumulators, stride {s}, {flag_text}",
           f":{p}",
           "mov rb11, unif",             # job list
           "mov ra20, unif",             # job count
           "shl r0, elem_num, 2",
           "mov rb12, r0",               # lane * 4
           "mov ra19, 16",
           "mov ra21, 64",
           f"mov ra22, {64 * s}",
           "mov ra23, 64",
           f":{p}_job",
           "mov unif_addr, rb11",
           "nop",
           "nop",
           "add rb0, unif, rb12",        # x (per lane)
           "mov rb1, unif",              # out0
           "mov rb2, unif",              # out1
           "add rb3, unif, rb12",        # R (per lane)
           "add rb4, unif, rb12",        # S (per lane)
           "mov ra16, unif",             # block count
           "mov rb5, unif",              # weight stream
           "mov rb6, unif",              # tap loop count
           "mov rb7, unif",              # row stride
           "mov rb8, unif",              # remaining valid input times
           "mov rb9, unif",              # alpha
           "mov ra22, unif" if flags & TM else "mov rb10, unif",  # scale (TM: output advance)
           "add rb11, rb11, ra23",
           f":{p}_block",
           "mov unif_addr, rb5",
           "mov r3, rb6",                # tap loop counter
           "nop",
           "add t0s, rb0, unif",         # taps 0 and 1
           "add t0s, rb0, unif"]
    out += [f"mov ra{a}, unif" for a in range(n)]  # bias
    out += ["mov r1, 0", f":{p}_tap"] + loop
    out += [f"fadd ra{n-1}, ra{n-1}, r1",
            "ldtmu0",                    # drop the 2 prefetches past the last tap
            "ldtmu0"]
    out += epilogue(n, s, flags)
    out += ["add rb0, rb0, ra21",
            "add rb1, rb1, ra22",
            "add rb2, rb2, ra22",
            "add rb3, rb3, ra21",
            "add rb4, rb4, ra21",
            "sub rb8, rb8, ra19",
            "sub.setf ra16, ra16, 1",
            f"brr.anynz -, :{p}_block",
            "nop",
            "nop",
            "nop",
            "sub.setf ra20, ra20, 1",
            f"brr.anynz -, :{p}_job",
            "nop",
            "nop",
            "nop",
            "thrend",
            "mov interrupt, 1",
            "nop"]
    check_hazards(out, p)
    check_hazards(loop + loop, p + " (tap loop wrap)")
    return out



# ---------------------------------------------------------------------------
# Flow: windowed relative-position attention, one head of 16 channels.

def f32(x):
    """A float constant as its exact bit pattern: vc4asm reads 1e-05 as an
    expression (1e - 05 = -4.0)."""
    return "0x%08x" % __import__("struct").unpack("<I", __import__("struct").pack("<f", x))[0]


KC = 16          # channels per head
WIN = 4          # relative-position window
ND = 2 * WIN + 1
# 2^f for f in (-1, 0]: Taylor series of e^(f ln 2), error < 2e-6.
EXP2_COEFFS = [0.6931471805599453 ** k / __import__("math").factorial(k) for k in range(8)]


def attention():
    """Job (16 words): 0 Q (head rows, first query), 1 Q row stride, 2 rel_k,
    3 keys before the band, 4 band keys, 5 keys after, 6 first band key - i0,
    7 K (time-major [T][16]), 8 KV (time-major [T][32]: K then V), 9 rel_v,
    10 remaining queries, 11 out (head rows, first query), 12 DMA stride word.

    One job = 16 queries i (one per lane). Q is pre-scaled by log2(e)/sqrt(16),
    so scores are in log2 units. Pass 1: m = max_j s_j. Pass 2: e_j =
    2^(s_j - m), sum += e_j, acc += e_j * V_j. s_j = q.K_j (+ q.rel_k[j-i]
    within the band |j-i| <= 4, done per lane with flags); the band's e_j
    are kept per d in P[d] for the rel_v term, added at the end:
    out = (acc + sum_d P[d] rel_v[d]) / sum."""
    q = [f"ra{c}" for c in range(KC)]
    rk = [f"ra{16 + d}" for d in range(ND)]       # d index 0..8 = offset -4..4
    acc = [f"rb{c}" for c in range(KC)]
    pd = [f"rb{16 + d}" for d in range(ND)]
    DV, ONE, M126, JOB, NJOBS, TI = "ra25", "ra26", "ra27", "ra28", "ra29", "ra30"
    JR, M, SUM, NBAND, F, NPOST, C23 = "rb25", "rb26", "rb27", "rb28", "rb29", "rb30", "rb31"

    def dot():  # r0 = q . (next 16 uniforms)
        ins = [f"fmul r0, {q[0]}, unif", f"fmul r1, {q[1]}, unif"]
        ins += [f"fadd r0, r0, r1; fmul r1, {q[c]}, unif" for c in range(2, KC)]
        return ins + ["fadd r0, r0, r1"]

    def band_rel_k():  # r0 += rel_k score for lanes with j - i == d
        ins = []
        for d in range(ND):
            ins += [f"sub.setf -, {DV}, {d - WIN}", f"fadd.ifz r0, r0, {rk[d]}"]
        return ins

    def exp2():  # r2 = 2^(r0 - m)
        ins = [f"fsub r2, r0, {M}",
               f"fmax r2, r2, {M126}",
               f"ftoi {TI}, r2",
               "nop",
               f"itof r1, {TI}",
               f"fsub {F}, r2, r1",                 # f = x - trunc(x), in (-1, 0]
               f"mov r1, {f32(EXP2_COEFFS[7])}"]
        for c in reversed(EXP2_COEFFS[1:7]):
            ins += [f"fmul r1, r1, {F}", f"mov r0, {f32(c)}", "fadd r1, r1, r0"]
        ins += [f"fmul r1, r1, {F}", "fadd r1, r1, 1.0",
                f"shl r0, {TI}, {C23}",             # * 2^trunc(x): add to the exponent
                "add r2, r1, r0"]
        return ins

    def body(pass_no, band):
        ins = []
        if band:
            ins += [f"sub {DV}, {JR}, elem_num", f"add {JR}, {JR}, {ONE}"]
        ins += dot()
        if band:
            ins += band_rel_k()
        if pass_no == 1:
            ins.append(f"fmax {M}, {M}, r0")
            return ins
        ins += exp2()
        ins.append(f"fadd {SUM}, {SUM}, r2")
        if band:
            for d in range(ND):
                ins += [f"sub.setf -, {DV}, {d - WIN}", f"mov.ifz {pd[d]}, r2"]
        ins.append("fmul r1, r2, unif")
        ins += [f"fadd {acc[c - 1]}, {acc[c - 1]}, r1; fmul r1, r2, unif" for c in range(1, KC)]
        ins.append(f"fadd {acc[KC - 1]}, {acc[KC - 1]}, r1")
        return ins

    def loop(label, count, ins):
        """Runs `ins` `count` times (count may be 0), counter in r3."""
        b = ins[:-3] + ["sub.setf r3, r3, 1"] + ins[-3:]
        b = b[:-3] + [f"brr.anynz -, :{label}"] + b[-3:]
        return ([f"mov.setf r3, {count}", f"brr.allz -, :{label}_end", "nop", "nop", "nop", f":{label}"] +
                b + [f":{label}_end"])

    def unif_at(word):  # rewind the uniform stream to job word `word`
        return [f"mov r0, {4 * word}", f"add r0, {JOB}, r0", "mov unif_addr, r0", "nop", "nop"]

    out = [":attn",
           f"mov {JOB}, unif",
           f"mov {NJOBS}, unif",
           f"mov {ONE}, 1",
           f"mov {M126}, {f32(-126.0)}",
           f"mov {C23}, 23",
           ":attn_job",
           f"mov unif_addr, {JOB}",
           "shl r1, elem_num, 2",
           "nop",
           "add r0, unif, r1",        # Q + 4 * lane
           "mov r3, unif",            # Q row stride
           "mov r2, unif",            # rel_k
           "mov unif_addr, r2"]
    for g in range(0, KC, 4):         # q -> registers (TMU FIFO: 4 requests)
        out += ["add r0, r0, r3; mov t0s, r0"] * 4
        for c in range(g, g + 4):
            out += ["ldtmu0", f"mov {q[c]}, r4"]
    for d in range(ND):               # rel_k scores per lane
        ins = dot()
        ins[-1] = f"fadd {rk[d]}, r0, r1"
        out += ins
    for pass_no in (1, 2):
        out += unif_at(3)
        out += ["mov r3, unif",       # keys before the band
                f"mov {NBAND}, unif",
                f"mov {NPOST}, unif",
                f"mov {JR}, unif",
                "mov r0, unif"]       # K
        if pass_no == 2:
            out.append("mov r0, unif")  # KV
        out.append("mov unif_addr, r0")
        if pass_no == 1:
            out += [f"mov {M}, {f32(-1e30)}", "nop"]
        else:
            out += [f"mov {r}, 0" for r in acc + pd + [SUM]]
        p = f"attn_p{pass_no}"
        out += loop(f"{p}_pre", "r3", body(pass_no, False))
        out += loop(f"{p}_band", NBAND, body(pass_no, True))
        out += loop(f"{p}_post", NPOST, body(pass_no, False))
    out += unif_at(9) + ["mov r0, unif", "mov unif_addr, r0", "nop", "nop"]
    for d in range(ND):               # + sum_d P[d] * rel_v[d]
        out += [f"mov r2, {pd[d]}", "fmul r1, r2, unif"]
        out += [f"fadd {acc[c - 1]}, {acc[c - 1]}, r1; fmul r1, r2, unif" for c in range(1, KC)]
        out.append(f"fadd {acc[KC - 1]}, {acc[KC - 1]}, r1")
    out += [f"mov sfu_recip, {SUM}", "nop", "nop",       # 1 / sum, 2 Newton steps
            f"fmul r0, {SUM}, r4", "fsub r0, 2.0, r0", "fmul r2, r4, r0",
            f"fmul r0, {SUM}, r2", "fsub r0, 2.0, r0", "fmul r2, r2, r0"]
    out += [f"fmul {a}, {a}, r2" for a in acc]
    out += unif_at(10) + ["mov r3, unif",      # remaining queries
                          "mov r0, unif",      # out
                          "mov r2, unif",      # DMA stride word
                          "sub.setf -, elem_num, r3"]
    out += [f"mov.ifnn {a}, 0" for a in acc]
    out += ["mov -, mutex", "mov vw_setup, vpm_setup(0, 1, h32(0))"]
    out += [f"mov vpm, {a}" for a in acc]
    out += [f"mov vw_setup, vdw_setup_0({KC}, 16, dma_h32(0, 0))",
            "mov vw_setup, r2",
            "mov vw_addr, r0",
            "mov -, vw_wait",
            "mov mutex, 0",
            "mov r0, 64",
            f"add {JOB}, {JOB}, r0",
            f"sub.setf {NJOBS}, {NJOBS}, 1",
            "brr.anynz -, :attn_job",
            "nop", "nop", "nop",
            "thrend", "mov interrupt, 1", "nop"]
    check_hazards(out, "attention")
    return out


# ---------------------------------------------------------------------------
# Flow: LayerNorm over 32 channels (ops::channelLayerNormInplace).

LN_C = 32


def layernorm():
    """Job (16 words): 0 x (first block), 1 out (first block), 2 row stride,
    3 block count, 4 gamma/beta stream (g0, b0, g1, b1, ...), 5 remaining
    times, 6 DMA stride word. Per block of 16 times: mean and variance over the
    32 channel rows, 1/sqrt(var + 1e-5) by the SFU plus 2 Newton steps."""
    x = [f"ra{c}" for c in range(16)] + [f"rb{c}" for c in range(16)]
    XA, OUT, STRIDE, GB, REM, DMA = "rb20", "rb21", "rb22", "rb23", "rb24", "rb25"
    JOB, NJOBS, NB, C16, C64 = "ra28", "ra29", "ra30", "ra27", "ra31"
    out = [":ln",
           f"mov {JOB}, unif",
           f"mov {NJOBS}, unif",
           f"mov {C16}, 16",
           f"mov {C64}, 64",
           ":ln_job",
           f"mov unif_addr, {JOB}",
           "shl r1, elem_num, 2",
           "nop",
           f"add {XA}, unif, r1",
           f"mov {OUT}, unif",
           f"mov {STRIDE}, unif",
           f"mov {NB}, unif",
           f"mov {GB}, unif",
           f"mov {REM}, unif",
           f"mov {DMA}, unif",
           "mov r0, 64",
           f"add {JOB}, {JOB}, r0",
           ":ln_block",
           f"mov unif_addr, {GB}",
           f"mov r0, {XA}"]
    for g in range(0, LN_C, 4):
        out += [f"add r0, r0, {STRIDE}; mov t0s, r0"] * 4
        for c in range(g, g + 4):
            out += ["ldtmu0", f"mov {x[c]}, r4"]
    out.append(f"fadd r0, {x[0]}, {x[16]}")
    for c in range(1, 16):
        out += [f"fadd r0, r0, {x[c]}", f"fadd r0, r0, {x[16 + c]}"]
    out += [f"mov r1, {f32(1.0 / LN_C)}", "fmul r0, r0, r1"]      # mean
    out += [f"fsub {x[c]}, {x[c]}, r0" for c in range(LN_C)]
    out += [f"fmul r2, {x[0]}, {x[0]}", f"fmul r1, {x[1]}, {x[1]}"]
    out += [f"fadd r2, r2, r1; fmul r1, {x[c]}, {x[c]}" for c in range(2, LN_C)]
    out += ["fadd r2, r2, r1",
            f"mov r1, {f32(1.0 / LN_C)}", "fmul r2, r2, r1",
            f"mov r1, {f32(1e-5)}", "fadd r2, r2, r1",                # var + eps
            "mov sfu_recipsqrt, r2", "nop", "nop",
            "mov r3, r4"]
    for _ in range(2):                                            # y *= 1.5 - 0.5 v y^2
        out += ["fmul r1, r3, r3", "fmul r1, r1, r2", "fmul r1, r1, 0.5",
                f"mov r0, {f32(1.5)}", "fsub r1, r0, r1", "fmul r3, r3, r1"]
    for c in range(LN_C):
        out += [f"fmul r1, {x[c]}, r3", "fmul r1, r1, unif", f"fadd {x[c]}, r1, unif"]
    out.append(f"sub.setf -, elem_num, {REM}")
    out += [f"mov.ifnn {x[c]}, 0" for c in range(LN_C)]
    out += ["mov -, mutex", "mov vw_setup, vpm_setup(0, 1, h32(0))"]
    out += [f"mov vpm, {x[c]}" for c in range(LN_C)]
    out += [f"mov vw_setup, vdw_setup_0({LN_C}, 16, dma_h32(0, 0))",
            f"mov vw_setup, {DMA}",
            f"mov vw_addr, {OUT}",
            "mov -, vw_wait",
            "mov mutex, 0",
            f"add {XA}, {XA}, {C64}",
            f"add {OUT}, {OUT}, {C64}",
            f"sub {REM}, {REM}, {C16}",
            f"sub.setf {NB}, {NB}, 1",
            "brr.anynz -, :ln_block",
            "nop", "nop", "nop",
            f"sub.setf {NJOBS}, {NJOBS}, 1",
            "brr.anynz -, :ln_job",
            "nop", "nop", "nop",
            "thrend", "mov interrupt, 1", "nop"]
    check_hazards(out, "layernorm")
    return out


REG = re.compile(r"\br[ab]\d+\b")


def check_hazards(lines, where):
    """A register file value written by one instruction can't be read by the next."""
    prev_writes = set()
    for line in lines:
        if line.startswith((":", "#")):
            continue
        writes, reads = set(), set()
        for part in line.split(";"):
            tokens = part.strip().split(None, 1)
            if len(tokens) < 2 or tokens[0].startswith(("brr", "ldtmu", "thrend", "nop")):
                continue
            operands = [o.strip() for o in tokens[1].split(",")]
            writes |= set(REG.findall(operands[0]))
            for o in operands[1:]:
                reads |= set(REG.findall(o))
        bad = reads & prev_writes
        if bad:
            raise ValueError(f"{where}: '{line}' reads {bad} written by the previous instruction")
        prev_writes = writes


def instructions(lines):
    return sum(1 for l in lines if l and not l.startswith((":", "#")))


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    kernels = [(KIND_CONV, n, s, flags, generate(n, s, flags)) for n, s, flags in VARIANTS]
    kernels += [(KIND_ATTENTION, KC, 1, 0, attention()), (KIND_LAYERNORM, LN_C, 1, 0, layernorm())]
    text, index, offset = [], [], 0
    for kind, n, s, flags, lines in kernels:
        index.append((kind, n, s, flags, offset))
        offset += instructions(lines)
        text += lines + [""]
    with open(os.path.join(out_dir, "qpu_kernels.qasm"), "w") as f:
        f.write("# Generated by research/rpi_qpu/gen_qpu_kernels.py -- do not edit.\n")
        f.write("\n".join(text))
    with open(os.path.join(out_dir, "qpu_kernels_index.h"), "w") as f:
        f.write("// Generated by research/rpi_qpu/gen_qpu_kernels.py -- do not edit.\n")
        f.write("// {kind (0 conv, 1 attention, 2 layernorm), accumulators, stride, flags,\n")
        f.write(f"// first instruction}} per kernel. {offset} instructions in total.\n")
        for kind, n, s, flags, off in index:
            f.write(f"{{{kind}, {n}, {s}, {flags}, {off}}},\n")
    print(f"{len(kernels)} kernels, {offset} instructions")
