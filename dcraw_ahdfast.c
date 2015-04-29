#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <inttypes.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
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
#define MOV_CLAMP(d,s) {(d)=(s); if(unlikely((s)<0)) (d)=0; else if(unlikely((s)>0xffff)) (d)=0xffff; }
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
void border_interpolate (int border);


extern const float d65_white[3];
extern const double xyz_rgb[3][3];
extern float rgb_cam[3][4];
extern unsigned colors;
float cielab_cbrt[0x10000], cielab_xyz_cam[3][4];
static void cielab_init()
{
    unsigned j;
    int i, k;
    float r;
    for (i=0; i < 0x10000; i++) {
        r = i / 65535.0;
        cielab_cbrt[i] = r > 0.008856 ? pow(r,1/3.0) : 7.787*r + 16/116.0;
    }
    for (i=0; i < 3; i++)
        for (j=0; j < colors; j++)
            for (cielab_xyz_cam[i][j] = k=0; k < 3; k++)
                cielab_xyz_cam[i][j] += xyz_rgb[i][k] * rgb_cam[k][j] / d65_white[i];
}

static void cielab (ushort rgb[3], short lab[3])
{
    unsigned c;
    float xyz[3];

    xyz[0] = xyz[1] = xyz[2] = 0.5;
    for(c=0; c<colors; c++) {
        xyz[0] += cielab_xyz_cam[0][c] * rgb[c];
        xyz[1] += cielab_xyz_cam[1][c] * rgb[c];
        xyz[2] += cielab_xyz_cam[2][c] * rgb[c];
    }
    if (unlikely(xyz[0]>0xffff)) xyz[0]=0xffff;
    else if (unlikely(xyz[0]<0)) xyz[0]=0;
    if (unlikely(xyz[1]>0xffff)) xyz[1]=0xffff;
    else if (unlikely(xyz[1]<0)) xyz[1]=0;
    if (unlikely(xyz[2]>0xffff)) xyz[2]=0xffff;
    else if (unlikely(xyz[2]<0)) xyz[2]=0;

    xyz[0] = cielab_cbrt[(int) xyz[0]];
    xyz[1] = cielab_cbrt[(int) xyz[1]];
    xyz[2] = cielab_cbrt[(int) xyz[2]];
    lab[0] = 64 * (116 * xyz[1] - 16);
    lab[1] = 64 * 500 * (xyz[0] - xyz[1]);
    lab[2] = 64 * 200 * (xyz[1] - xyz[2]);
}




union rgbpix {
    uint64_t P;
    ushort c[4];
    struct {
        ushort r;
        ushort g;
        ushort b;
        ushort g1;
    };
};

union hvrgbpix {
    struct {
        union rgbpix h;
        union rgbpix v;
    };
};

void ahd_interpolate_tile(int top, char * buffer)
{
    int i, j, row, col, tr, tc, c, d, val, hm[2];
    const int dir[4] = { -1, 1, -width, width };
    unsigned ldiff[2][4], abdiff[2][4], leps, abeps;
    union hvrgbpix (*rgb)[width] = (union hvrgbpix (*)[width])buffer;
    union hvrgbpix *rix;
    union rgbpix * pix;
    short (*lab)[TS][width][8], (*lix)[8];
    char (*homo)[TS][width];
    lab  = (short (*)[TS][width][8])(buffer + 16*width*TS);
    homo = (char  (*)[TS][width])(buffer + 32*width*TS);

    const int left=2;

    if ((uintptr_t)(image+top*width)&0xf || (uintptr_t)buffer&0xf) {
        fprintf(stderr, "unaligned buffers defeat speed!\n"); abort();
    }

    /*  Interpolate gren horz&vert, red and blue, and convert to CIELab:  */
    //do the first two rows of green first.
    //then one green, and rgb through the tile.. this because R/B needs down-right green value
    for (row=top; row < top+2 && row < height-2; row++) {
        col = left + (FC(row,left) & 1);
        for (c = FC(row,col); col < width-2; col+=2) {
            pix = (union rgbpix*)image + row*width+col;
            val = ((pix[-1].g + pix[0].c[c] + pix[1].g) * 2 - pix[-2].c[c] - pix[2].c[c]) >> 2;
            rgb[row-top][col-left].h.g = ULIM(val,pix[-1].g,pix[1].g);
            val = ((pix[-width].g + pix[0].c[c] + pix[width].g) * 2 - pix[-2*width].c[c] - pix[2*width].c[c]) >> 2;
            rgb[row-top][col-left].v.g = ULIM(val,pix[-width].g,pix[width].g);
        }
    }

    for (; row < top+TS && row < height-2; row++) {
        int rowx = row-1;

        if (FC(rowx,left+1)==1) {
            int c1 = FC(rowx+1,left+1),
                c2 = FC(rowx,left+2);

            pix = (union rgbpix*)image + row*width+left+1;
            val = ((pix[-1].g + pix[0].c[c1] + pix[1].g) * 2 - pix[-2].c[c1] - pix[2].c[c1]) >> 2;
            rgb[row-top][1].h.g = ULIM(val,pix[-1].g,pix[1].g);
            val = ((pix[-width].g + pix[0].c[c1] + pix[width].g) * 2 - pix[-2*width].c[c1] - pix[2*width].c[c1]) >> 2;
            rgb[row-top][1].v.g = ULIM(val,pix[-width].g,pix[width].g);
            for (col=left+1; col < width-3; col+=2) {
                pix = (union rgbpix*)image + rowx*width+col;

                union rgbpix rix0_0,rix0_r;
                union rgbpix rix1_0,rix1_r;

                rix = &rgb[rowx-top][col-left];

                signed pix_ud = pix[-width].c[c1] + pix[width].c[c1];
                signed pix_diag = pix[-width].c[c1] + pix[-width+2].c[c1];
                signed pix_lr = pix[-1].c[c2] + pix[1].c[c2];
                rix1_r.c[c2] = rix0_r.c[c2]  = pix[1].c[c2];
                rix1_0.g = rix0_0.g = pix[0].g;
                pix_diag += pix[width].c[c1] + pix[width+2].c[c1];
                val = ((pix[width+1].g + pix[width+2].c[c1] + pix[width+3].g) * 2 - pix[width].c[c1] - pix[width+4].c[c1]) >> 2;
                signed rix0_dr = ULIM(val,pix[width+1].g,pix[width+3].g);
                val = ((pix[2].g + pix[width+2].c[c1] + pix[2*width+2].g) * 2 - pix[-width+2].c[c1] - pix[3*width+2].c[c1]) >> 2;
                signed rix1_dr = ULIM(val,pix[2].g,pix[2*width+2].g);


                signed rix0_u = rix[-width].h.g;
                signed rix1_u = rix[-width].v.g;
                signed rix0_ur = rix[-width+2].h.g;
                signed rix1_ur = rix[-width+2].v.g;
                signed rix0_l = rix[-1].h.g;
                signed rix1_l = rix[-1].v.g;
                rix0_r.g = rix[1].h.g;
                rix1_r.g = rix[1].v.g;
                signed rix0_d = rix[width].h.g;
                signed rix1_d = rix[width].v.g;

                signed rix0_ud = rix0_u + rix0_d;
                signed rix1_ud = rix1_u + rix1_d;

                val = rix0_0.g + (( pix_lr - rix0_l - rix0_r.g ) >> 1);
                MOV_CLAMP(rix0_0.c[c2],val);
                val = rix0_0.g + (( pix_lr - rix1_l - rix1_r.g) >> 1);
                MOV_CLAMP(rix1_0.c[c2], val);
                val = rix0_0.g + (( pix_ud - rix0_ud ) >> 1);
                MOV_CLAMP(rix0_0.c[c1],val);
                val = rix0_0.g + (( pix_ud - rix1_ud ) >> 1);
                MOV_CLAMP(rix1_0.c[c1],val);

                val = rix0_r.g + (( pix_diag - rix0_ud - rix0_ur - rix0_dr + 1) >> 2);
                MOV_CLAMP(rix0_r.c[c1],val);
                val = rix1_r.g + (( pix_diag - rix1_ud - rix1_ur - rix1_dr + 1) >> 2);
                MOV_CLAMP(rix1_r.c[c1],val);

                cielab (rix0_0.c,lab[0][rowx-top][col-left]);
                cielab (rix1_0.c,&lab[0][rowx-top][col-left][4]);
                cielab (rix0_r.c,lab[0][rowx-top][col-left+1]);
                cielab (rix1_r.c,&lab[0][rowx-top][col-left+1][4]);
                memcpy(&rix[0].h,&rix0_0,6);
                memcpy(&rix[0].v,&rix1_0,6);
                memcpy(&rix[1].h,&rix0_r,6);
                memcpy(&rix[1].v,&rix1_r,6);
                rix[width+2].h.g = rix0_dr;
                rix[width+2].v.g = rix1_dr;
            }
        } else {
            int c1 = FC(rowx,left+1),
                c2 = FC(rowx+1,left+2);
            pix = (union rgbpix*)image + row*width+left;
            val = ((pix[-1].g + pix[0].c[c2] + pix[1].g) * 2 - pix[-2].c[c2] - pix[2].c[c2]) >> 2;
            rgb[row-top][0].h.g = ULIM(val,pix[-1].g,pix[1].g);
            val = ((pix[-width].g + pix[0].c[c2] + pix[width].g) * 2 - pix[-2*width].c[c2] - pix[2*width].c[c2]) >> 2;
            rgb[row-top][0].v.g = ULIM(val,pix[-width].g,pix[width].g);


            for (col=left+1; col < width-3; col+=2) {
                pix = (union rgbpix*)image + rowx*width+col;

                union rgbpix rix0_0,rix0_r;
                union rgbpix rix1_0,rix1_r;

                rix = &rgb[rowx-top][col-left];

                signed pix_diag = pix[-width-1].c[c2] + pix[-width+1].c[c2];
                signed pix_ur = pix[-width+1].c[c2];
                rix1_0.c[c1] = rix0_0.c[c1] = pix[0].c[c1];
                signed pix_lr = pix[0].c[c1] + pix[2].c[c1];
                rix1_r.g = rix0_r.g = pix[1].g;
                pix_diag += pix[width-1].c[c2] + pix[width+1].c[c2];
                signed pix_dr = pix[width+1].c[c2];

                val = ((pix[width].g + pix[width+1].c[c2] + pix[width+2].g) * 2 - pix[width-1].c[c2] - pix[width+3].c[c2]) >> 2;
                signed rix0_dr = ULIM(val,pix[width].g,pix[width+2].g);
                val = ((rix1_r.g + pix_dr + pix[2*width+1].g) * 2 - pix_ur - pix[3*width+1].c[c2]) >> 2;
                signed rix1_dr = ULIM(val,pix[1].g,pix[2*width+1].g);

                signed pix_udr = pix_dr+pix_ur;

                signed rix0_ul = rix[-width-1].h.g;
                signed rix1_ul = rix[-width-1].v.g;
                signed rix0_ur = rix[-width+1].h.g;
                signed rix1_ur = rix[-width+1].v.g;
                rix0_0.g = rix[0].h.g;
                rix1_0.g = rix[0].v.g;
                signed rix0_rr = rix[2].h.g;
                signed rix1_rr = rix[2].v.g;
                signed rix0_dl = rix[width-1].h.g;
                signed rix1_dl = rix[width-1].v.g;

                signed rix0_udr = rix0_dr+rix0_ur;
                signed rix1_udr = rix1_dr+rix1_ur;


                val = rix0_r.g + (( pix_lr - rix0_0.g - rix0_rr ) >> 1);
                MOV_CLAMP(rix0_r.c[c1], val);
                val = rix0_r.g + (( pix_lr - rix1_0.g - rix1_rr ) >> 1);
                MOV_CLAMP(rix1_r.c[c1], val);
                val = rix0_r.g + ((pix_udr - rix0_udr ) >> 1);
                MOV_CLAMP(rix0_r.c[c2], val);
                val = rix0_r.g + ((pix_udr - rix1_udr ) >> 1);
                MOV_CLAMP(rix1_r.c[c2], val);

                val = rix0_0.g + ((pix_diag - rix0_ul - rix0_dl - rix0_udr + 1) >> 2);
                MOV_CLAMP(rix0_0.c[c2],val);
                val = rix1_0.g + ((pix_diag - rix1_ul - rix1_udr - rix1_dl + 1) >> 2);
                MOV_CLAMP(rix1_0.c[c2],val);


                cielab (rix0_0.c,lab[0][rowx-top][col-left]);
                cielab (rix1_0.c,&lab[0][rowx-top][col-left][4]);
                cielab (rix0_r.c,lab[0][rowx-top][col-left+1]);
                cielab (rix1_r.c,&lab[0][rowx-top][col-left+1][4]);
                memcpy(&rix[0].h,&rix0_0,6);
                memcpy(&rix[0].v,&rix1_0,6);
                memcpy(&rix[1].h,&rix0_r,6);
                memcpy(&rix[1].v,&rix1_r,6);
                rix[width+1].h.g = rix0_dr;
                rix[width+1].v.g = rix1_dr;
            }
        }
    }
/*  Build homogeneity maps from the CIELab images:    */
    memset (homo, 0, 2*width*TS);
    for (row=top+2; row < top+TS-2 && row < height-4; row++) {
        tr = row-top;
        for (col=left+2; col < width-4; col++) {
            tc = col-left;
            lix = &lab[0][tr][tc];
            for (i=0; i < 4; i++) {
                ldiff[0][i] = ABS(lix[0][0]-lix[dir[i]][0]);
                abdiff[0][i] = SQR(lix[0][1]-lix[dir[i]][1])
                               + SQR(lix[0][2]-lix[dir[i]][2]);
                ldiff[1][i] = ABS(lix[0][0+4]-lix[dir[i]][0+4]);
                abdiff[1][i] = SQR(lix[0][1+4]-lix[dir[i]][1+4])
                               + SQR(lix[0][2+4]-lix[dir[i]][2+4]);
            }
            leps = MIN(MAX(ldiff[0][0],ldiff[0][1]),
                       MAX(ldiff[1][2],ldiff[1][3]));
            abeps = MIN(MAX(abdiff[0][0],abdiff[0][1]),
                        MAX(abdiff[1][2],abdiff[1][3]));
            for (d=0; d < 2; d++)
                homo[d][tr][tc]+=
                    (ldiff[d][0] <= leps && abdiff[d][0] <= abeps)+
                    (ldiff[d][1] <= leps && abdiff[d][1] <= abeps)+
                    (ldiff[d][2] <= leps && abdiff[d][2] <= abeps)+
                    (ldiff[d][3] <= leps && abdiff[d][3] <= abeps);


        }
    }
/*  Combine the most homogenous pixels for the final result:  */
    for (row=top+3; row < top+TS-3 && row < height-5; row++) {
        tr = row-top;
        for (col=left+3; col < width-5; col++) {
            tc = col-left;
            for (d=0; d < 2; d++)
                for (hm[d]=0, i=tr-1; i <= tr+1; i++)
                    for (j=tc-1; j <= tc+1; j++)
                        hm[d] += homo[d][i][j];
            if (hm[0] != hm[1])
                FORC3 image[row*width+col][c] = rgb[tr][tc].h.c[c + (hm[1] > hm[0])*4];
            else
                FORC3 image[row*width+col][c] = (rgb[tr][tc].h.c[c] + rgb[tr][tc].v.c[c]) >> 1;
        }
    }
}


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

#define NUM_CHUNKS 4
void * ahd_interpolate_worker(void*args) {
    unsigned top;
    unsigned chunk = (unsigned)(uintptr_t)args;
    char * buffer = (char *) malloc (34*width*TS);

    if (!buffer) {
        fprintf(stderr, "memory alloc error, %s\n",__FUNCTION__);
        return NULL;
    }

    if (chunk==0) top=2;
    else top = height/NUM_CHUNKS * chunk;

    for (; top < height/NUM_CHUNKS*(chunk+1); top += TS-6)
        ahd_interpolate_tile(top,buffer);

    free (buffer);
    return NULL;
}

void ahd_interpolate_fast(void)
{
#ifdef THREADED
    pthread_t th[NUM_CHUNKS-1];
    pthread_attr_t tattr;
#endif
    struct timespec start;

    if (verbose) fprintf (stderr,_("AHD interpolation...\n"));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    cielab_init();
    border_interpolate(5);

#ifdef THREADED
    pthread_attr_init(&tattr);
    pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);
#endif
    for(unsigned i=0; i<NUM_CHUNKS-1; i++) {
#ifdef THREADED
        pthread_create(&th[i], &tattr, ahd_interpolate_worker, (void*)(uintptr_t)i);
#else
        ahd_interpolate_worker((void*)(uintptr_t)i);
#endif
    }
    ahd_interpolate_worker((void*)NUM_CHUNKS-1);
#ifdef THREADED
    pthread_attr_destroy(&tattr);
    for(unsigned i=0; i<(sizeof th/sizeof th[0]) -1; i++)
        pthread_join(th[i],NULL);
#endif

    fprintf(stderr, "%15s %12" PRIu64 " us (1926200)\n","runtime",timediff(&start));
}
