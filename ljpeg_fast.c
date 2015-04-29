#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

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

extern FILE *ifp;
extern unsigned zero_after_ff, dng_version;


static unsigned getbithuff (int nbits, ushort *huff)
{
    static unsigned bitbuf=0;
    static int vbits=0, reset=0;
    unsigned c;

    if (nbits > 25) return 0;
    if (nbits < 0)
        return bitbuf = vbits = reset = 0;
    if (nbits == 0 || vbits < 0) return 0;
    while (!reset && vbits < nbits && (c = fgetc(ifp)) != (unsigned)EOF &&
           !(reset = zero_after_ff && c == 0xff && fgetc(ifp))) {
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
            fseek (ifp, -2, SEEK_CUR);
            do mark = (mark << 8) + (c = fgetc(ifp));
            while (c != EOF && mark >> 4 != 0xffd);
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

void ljpeg_end_fast (struct jhead *jh)
{
    int c;
    for(c=0; c<4; c++) if (jh->free[c]) free (jh->free[c]);
    free (jh->row);
}

int ljpeg_start_fast (struct jhead *jh, int info_only)
{
    int c, tag, len;
    uchar data[0x10000];
    const uchar *dp;
    size_t fread_b;

    memset (jh, 0, sizeof *jh);
    jh->restart = INT_MAX;
    fread_b=fread (data, 2, 1, ifp);
    if(fread_b!=1) derror();
    if (data[1] != 0xd8) return 0;
    do {
        fread_b = fread (data, 2, 2, ifp);
        if (fread_b != 2) derror();
        tag =  data[0] << 8 | data[1];
        len = (data[2] << 8 | data[3]) - 2;
        if (tag <= 0xff00) return 0;
        fread_b=fread (data, 1, len, ifp);
        if(fread_b!=(unsigned)len) derror();
        switch (tag) {
        case 0xffc3:
            jh->sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
        case 0xffc0:
            jh->bits = data[0];
            jh->high = data[1] << 8 | data[2];
            jh->wide = data[3] << 8 | data[4];
            jh->clrs = data[5] + jh->sraw;
            if (len == 9 && !dng_version) getc(ifp);
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
