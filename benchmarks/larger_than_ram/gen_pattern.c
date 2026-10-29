// gen_pattern.c -- deterministic, position-dependent backing file for the
// VRAMER "larger than RAM" disk-tier benchmark.
//
// Every 4 KiB block b is filled with the single byte (b & 0xff), where b is the
// block's absolute index from the start of the file. This mirrors the
// per-page-seed convention AnonymousRamSource already uses ((i/page)&0xff) and
// lets the bench's sampling checksum be predicted in closed form:
//
//   vramer bench's sample_kernel sums one byte per 4096-byte block over the
//   whole scan (compute-reps=1). With this pattern the sampled byte of block b
//   is (b & 0xff), so for a file of M MiB (a multiple of 1 MiB) the expected
//   checksum is exactly:
//
//       checksum = M * 32640          (32640 = sum_{k=0}^{255} k, one full MiB)
//
//   This is INDEPENDENT of the pager's --page-size (any page size that is a
//   multiple of 4096 and tiles the file produces the identical checksum), which
//   makes it a strong cross-configuration consistency check.
//
// Usage: gen_pattern <path> <size_in_MiB>
// The size must be a whole number of MiB (1 MiB = 256 blocks of 4 KiB).
//
// Build: cc -O2 -o gen_pattern gen_pattern.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <inttypes.h>

#define MiB (1024u * 1024u)
#define BLOCK 4096u

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <path> <size_in_MiB>\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    uint64_t mib = strtoull(argv[2], NULL, 10);
    if (mib == 0) { fprintf(stderr, "size must be > 0 MiB\n"); return 2; }

    // One 1 MiB pattern buffer. The pattern has period 1 MiB (256 blocks ->
    // byte values 0..255), and every 1 MiB chunk of the file starts on a block
    // index that is a multiple of 256, so the same buffer is written verbatim
    // for every MiB -- fast and allocation-light.
    unsigned char* buf = malloc(MiB);
    if (!buf) { perror("malloc"); return 1; }
    for (uint32_t b = 0; b < MiB / BLOCK; ++b)        // 256 blocks per MiB
        memset(buf + (size_t)b * BLOCK, (int)(b & 0xff), BLOCK);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }

    for (uint64_t i = 0; i < mib; ++i) {
        size_t off = 0;
        while (off < MiB) {
            ssize_t w = write(fd, buf + off, MiB - off);
            if (w < 0) { perror("write"); return 1; }
            off += (size_t)w;
        }
    }
    if (fsync(fd) != 0) { perror("fsync"); return 1; }
    close(fd);
    free(buf);

    uint64_t checksum = mib * 32640ull;  // see header comment
    printf("wrote %" PRIu64 " MiB (%.2f GiB) to %s\n",
           mib, (double)mib / 1024.0, path);
    printf("expected vramer-bench checksum (compute-reps=1): %" PRIu64 "\n", checksum);
    return 0;
}
