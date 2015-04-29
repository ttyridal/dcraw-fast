#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <inttypes.h>
#include <time.h>
#if !defined(ushort)
#define ushort unsigned short
#endif
#define TS 512                /* Tile Size */
#define SQR(x) ((x)*(x))
#define ABS(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define LIM(x,min,max) MAX(min,MIN(x,max))
#define ULIM(x,y,z) ((y) < (z) ? LIM(x,y,z) : LIM(x,z,y))
#define CLIP(x) LIM(x,0,65535)
#define FORC(cnt) for (c=0; c < cnt; c++)
#define FORC3 FORC(3)
#define FC(row,col) \
    (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3)
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

extern const ushort height,width;
extern ushort (*image)[4];
extern const unsigned filters;

extern const int verbose;
#define _(x) x
void merror (void *ptr, const char *where);
void border_interpolate (int border);

void cielab (ushort rgb[3], short lab[3]);



/*
   Adaptive Homogeneity-Directed interpolation is based on
   the work of Keigo Hirakawa, Thomas Parks, and Paul Lee.
 */
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
void ahd_interpolate_fast(void)
{
    int i, j, top, left, row, col, tr, tc, c, d, val, hm[2];
    static const int dir[4] = { -1, 1, -TS, TS };
    unsigned ldiff[2][4], abdiff[2][4], leps, abeps;
    ushort (*rgb)[TS][TS][3], (*rix)[3], (*pix)[4];
    short (*lab)[TS][TS][3], (*lix)[3];
    char (*homo)[TS][TS], *buffer;
    struct timespec start;

    if (verbose) fprintf (stderr,_("AHD interpolation...\n"));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    cielab (0,0);
    border_interpolate(5);
    buffer = (char *) malloc (26*TS*TS);
    merror (buffer, "ahd_interpolate()");
    rgb  = (ushort(*)[TS][TS][3])buffer;
    lab  = (short (*)[TS][TS][3])(buffer + 12*TS*TS);
    homo = (char  (*)[TS][TS])(buffer + 24*TS*TS);

    for (top=2; top < height-5; top += TS-6)
        for (left=2; left < width-5; left += TS-6) {

/*  Interpolate green horizontally and vertically:    */
            for (row=top; row < top+TS && row < height-2; row++) {
                col = left + (FC(row,left) & 1);
                for (c = FC(row,col); col < left+TS && col < width-2; col+=2) {
                    pix = image + row*width+col;
                    val = ((pix[-1][1] + pix[0][c] + pix[1][1]) * 2
                           - pix[-2][c] - pix[2][c]) >> 2;
                    rgb[0][row-top][col-left][1] = ULIM(val,pix[-1][1],pix[1][1]);
                    val = ((pix[-width][1] + pix[0][c] + pix[width][1]) * 2
                           - pix[-2*width][c] - pix[2*width][c]) >> 2;
                    rgb[1][row-top][col-left][1] = ULIM(val,pix[-width][1],pix[width][1]);
                }
            }
/*  Interpolate red and blue, and convert to CIELab:    */
            for (d=0; d < 2; d++)
                for (row=top+1; row < top+TS-1 && row < height-3; row++)
                    for (col=left+1; col < left+TS-1 && col < width-3; col++) {
                        pix = image + row*width+col;
                        rix = &rgb[d][row-top][col-left];
                        lix = &lab[d][row-top][col-left];
                        if ((c = 2 - FC(row,col)) == 1) {
                            c = FC(row+1,col);
                            val = pix[0][1] + (( pix[-1][2-c] + pix[1][2-c]
                                                 - rix[-1][1] - rix[1][1] ) >> 1);
                            rix[0][2-c] = CLIP(val);
                            val = pix[0][1] + (( pix[-width][c] + pix[width][c]
                                                 - rix[-TS][1] - rix[TS][1] ) >> 1);
                        } else
                            val = rix[0][1] + (( pix[-width-1][c] + pix[-width+1][c]
                                                 + pix[+width-1][c] + pix[+width+1][c]
                                                 - rix[-TS-1][1] - rix[-TS+1][1]
                                                 - rix[+TS-1][1] - rix[+TS+1][1] + 1) >> 2);
                        rix[0][c] = CLIP(val);
                        c = FC(row,col);
                        rix[0][c] = pix[0][c];
                        cielab (rix[0],lix[0]);
                    }
/*  Build homogeneity maps from the CIELab images:    */
            memset (homo, 0, 2*TS*TS);
            for (row=top+2; row < top+TS-2 && row < height-4; row++) {
                tr = row-top;
                for (col=left+2; col < left+TS-2 && col < width-4; col++) {
                    tc = col-left;
                    for (d=0; d < 2; d++) {
                        lix = &lab[d][tr][tc];
                        for (i=0; i < 4; i++) {
                            ldiff[d][i] = ABS(lix[0][0]-lix[dir[i]][0]);
                            abdiff[d][i] = SQR(lix[0][1]-lix[dir[i]][1])
                                           + SQR(lix[0][2]-lix[dir[i]][2]);
                        }
                    }
                    leps = MIN(MAX(ldiff[0][0],ldiff[0][1]),
                               MAX(ldiff[1][2],ldiff[1][3]));
                    abeps = MIN(MAX(abdiff[0][0],abdiff[0][1]),
                                MAX(abdiff[1][2],abdiff[1][3]));
                    for (d=0; d < 2; d++)
                        for (i=0; i < 4; i++)
                            if (ldiff[d][i] <= leps && abdiff[d][i] <= abeps)
                                homo[d][tr][tc]++;
                }
            }
/*  Combine the most homogenous pixels for the final result:  */
            for (row=top+3; row < top+TS-3 && row < height-5; row++) {
                tr = row-top;
                for (col=left+3; col < left+TS-3 && col < width-5; col++) {
                    tc = col-left;
                    for (d=0; d < 2; d++)
                        for (hm[d]=0, i=tr-1; i <= tr+1; i++)
                            for (j=tc-1; j <= tc+1; j++)
                                hm[d] += homo[d][i][j];
                    if (hm[0] != hm[1])
                        FORC3 image[row*width+col][c] = rgb[hm[1] > hm[0]][tr][tc][c];
                    else
                        FORC3 image[row*width+col][c] =
                            (rgb[0][tr][tc][c] + rgb[1][tr][tc][c]) >> 1;
                }
            }
        }
    free (buffer);
    fprintf(stderr, "%15s %12" PRIu64 " us (1926200)\n","runtime",timediff(&start));

}

