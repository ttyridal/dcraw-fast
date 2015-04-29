#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>

#if !defined(ushort)
#define ushort unsigned short
#endif
#if !defined(uchar)
#define uchar unsigned char
#endif

void derror();
void merror (void *ptr, const char *where);
ushort * make_decoder_ref (const uchar **source);

struct jhead {
    int bits, high, wide, clrs, sraw, psv, restart, vpred[6];
    ushort *huff[6], *free[4], *row;
};
unsigned char * ifp_m;
size_t pos;
size_t max_pos;

extern FILE *ifp;
extern unsigned zero_after_ff, dng_version;

static uint64_t timediff(const struct timespec * start)
{
    struct timespec end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    struct timespec temp;
    if ((end.tv_nsec-start->tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start->tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start->tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start->tv_sec;
        temp.tv_nsec = end.tv_nsec-start->tv_nsec;
    }
    return temp.tv_sec*1000000 + temp.tv_nsec/1000;
}

static unsigned getbithuff (int nbits, ushort *huff)
{
    static unsigned bitbuf=0;
    static int vbits=0, reset=0;
    unsigned c;

    if (nbits > 25) return 0;
    if (nbits < 0)
        return bitbuf = vbits = reset = 0;
    if (nbits == 0 || vbits < 0) return 0;

    while (  !reset
            && vbits < nbits
            && pos<max_pos
            && (c = ifp_m[pos++],1)
            && !(
                reset = zero_after_ff
                && c == 0xff && ifp_m[pos++]
               )
       ) {
        bitbuf = (bitbuf << 8) + (uchar) c;
        vbits += 8;
    }
    c = bitbuf << (32-vbits) >> (32-nbits);
    if (huff) {
        vbits -= huff[c] >> 8;
        c = (uchar) huff[c];
    } else
        vbits -= nbits;
    if (vbits < 0) derror();
    return c;
}

#define getbits(n) getbithuff(n,0)
#define gethuff(h) getbithuff(*h,h+1)



static int ljpeg_diff (ushort *huff)
{
    int len, diff;

    len = gethuff(huff);
    if (len == 16 && (!dng_version || dng_version >= 0x1010000))
        return -32768;
    diff = getbits(len);
    if ((diff & (1 << (len-1))) == 0)
        diff -= (1 << len) - 1;
    return diff;
}

ushort * ljpeg_row_fast (int jrow, struct jhead *jh)
{
    int col, c, diff, pred, spred=0;
    ushort mark=0, *row[3];

    if (jrow * jh->wide % jh->restart == 0) {
        for(c=0; c<6; c++) jh->vpred[c] = 1 << (jh->bits-1);
        if (jrow) {
            pos-=2;
            do {
                mark = (mark << 8) + ifp_m[pos++];
            }
            while (pos<max_pos && mark >> 4 != 0xffd);
        }
        getbits(-1);
    }
    for(c=0; c<3; c++) row[c] = jh->row + jh->wide*jh->clrs*((jrow+c) & 1);
    for (col=0; col < jh->wide; col++)
        for(c=0; c<jh->clrs; c++) {
            diff = ljpeg_diff (jh->huff[c]);
            if (jh->sraw && c <= jh->sraw && (col | c))
                pred = spred;
            else if (col) pred = row[0][-jh->clrs];
            else pred = (jh->vpred[c] += diff) - diff;
            if (jrow && col) switch (jh->psv) {
                case 1:
                    break;
                case 2:
                    pred = row[1][0];
                    break;
                case 3:
                    pred = row[1][-jh->clrs];
                    break;
                case 4:
                    pred = pred +   row[1][0] - row[1][-jh->clrs];
                    break;
                case 5:
                    pred = pred + ((row[1][0] - row[1][-jh->clrs]) >> 1);
                    break;
                case 6:
                    pred = row[1][0] + ((pred - row[1][-jh->clrs]) >> 1);
                    break;
                case 7:
                    pred = (pred + row[1][0]) >> 1;
                    break;
                default:
                    pred = 0;
                }
            if ((**row = pred + diff) >> jh->bits) derror();
            if (c <= jh->sraw) spred = **row;
            row[0]++; row[1]++;
        }
    return row[2];
}

static struct timespec start;

void ljpeg_end_fast (struct jhead *jh)
{
    int c;
    for(c=0; c<4; c++) if (jh->free[c]) free (jh->free[c]);
    munmap(ifp_m,0);
    free (jh->row);
//     fseek(ifp, pos, SEEK_SET);
    fprintf(stderr, "%15s %12" PRIu64 " us\n","runtime (jlpeg)",timediff(&start));
}

int ljpeg_start_fast (struct jhead *jh, int info_only)
{
    int c, tag, len;
    uchar data[0x10000];
    const uchar *dp;


    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    memset (jh, 0, sizeof *jh);
    pos = ftell(ifp);
    fseek(ifp, 0, SEEK_END);
    max_pos = ftell(ifp);
    ifp_m = mmap(NULL, max_pos, PROT_READ,MAP_SHARED, fileno(ifp), 0);

    jh->restart = INT_MAX;
    pos+=2;
    if (ifp_m[pos-1] != 0xd8) return 0;
    do {
//         memcpy(data, ifp_m+pos,4); pos+=4;
        pos+=4;
        tag =  ifp_m[pos-4] << 8 | ifp_m[pos-3];
        len = (ifp_m[pos-2] << 8 | ifp_m[pos-1]) - 2;
        if (tag <= 0xff00) return 0;
        memcpy(data, ifp_m+pos,len); pos+=len;
        switch (tag) {
        case 0xffc3:
            jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
        case 0xffc0:
            jh->bits = data[0];
            jh->high = data[1] << 8 | data[2];
            jh->wide = data[3] << 8 | data[4];
            jh->clrs = data[5] + jh->sraw;
            if (len == 9 && !dng_version) pos++;
            break;
        case 0xffc4:
            if (info_only) break;
            for (dp = data; dp < data+len && (c = *dp++) < 4; )
                jh->free[c] = jh->huff[c] = make_decoder_ref (&dp);
            break;
        case 0xffda:
            jh->psv = data[1+data[0]*2];
            jh->bits -= data[3+data[0]*2] & 15;
            break;
        case 0xffdd:
            jh->restart = data[0] << 8 | data[1];
        }
    } while (tag != 0xffda);
    if (info_only) return 1;
    if (jh->clrs > 6 || !jh->huff[0]) return 0;
    for(c=0; c<5; c++) if (!jh->huff[c+1]) jh->huff[c+1] = jh->huff[c];
    if (jh->sraw) {
        for(c=0; c<4; c++) {
            jh->huff[2+c] = jh->huff[1];
        }
        for(c=0; c<jh->sraw; c++) {
            jh->huff[2+c] = jh->huff[0];
        }
    }
    jh->row = (ushort *) calloc (jh->wide*jh->clrs, 4);
    merror (jh->row, "ljpeg_start()");
    return zero_after_ff = 1;
}
