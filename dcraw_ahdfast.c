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
    __m128i vec;
    struct {
        union rgbpix h;
        union rgbpix v;
    };
};

static __m128i cielabv (union hvrgbpix rgb)
{
    __m128 xvxyz[2] = {_mm_set1_ps(0.5),_mm_set1_ps(0.5) }; //,0.5,0.5,0.5);

    __m128 vcam0 = _mm_setr_ps(cielab_xyz_cam[0][0],cielab_xyz_cam[1][0],cielab_xyz_cam[2][0],0);
    __m128 vcam1 = _mm_setr_ps(cielab_xyz_cam[0][1],cielab_xyz_cam[1][1],cielab_xyz_cam[2][1],0);
    __m128 vcam2 = _mm_setr_ps(cielab_xyz_cam[0][2],cielab_xyz_cam[1][2],cielab_xyz_cam[2][2],0);
    __m128 vrgb0h = _mm_set1_ps(rgb.h.c[0]);
    __m128 vrgb1h = _mm_set1_ps(rgb.h.c[1]);
    __m128 vrgb2h = _mm_set1_ps(rgb.h.c[2]);
    __m128 vrgb0v = _mm_set1_ps(rgb.v.c[0]);
    __m128 vrgb1v = _mm_set1_ps(rgb.v.c[1]);
    __m128 vrgb2v = _mm_set1_ps(rgb.v.c[2]);

    xvxyz[0] = _mm_add_ps(xvxyz[0], _mm_mul_ps(vcam0,vrgb0h));
    xvxyz[0] = _mm_add_ps(xvxyz[0], _mm_mul_ps(vcam1,vrgb1h));
    xvxyz[0] = _mm_add_ps(xvxyz[0], _mm_mul_ps(vcam2,vrgb2h));
    xvxyz[1] = _mm_add_ps(xvxyz[1], _mm_mul_ps(vcam0,vrgb0v));
    xvxyz[1] = _mm_add_ps(xvxyz[1], _mm_mul_ps(vcam1,vrgb1v));
    xvxyz[1] = _mm_add_ps(xvxyz[1], _mm_mul_ps(vcam2,vrgb2v));

    xvxyz[0] = _mm_max_ps(_mm_set1_ps(0),
                          _mm_min_ps(_mm_set1_ps(0xffff),
                                     _mm_round_ps(xvxyz[0], _MM_FROUND_TO_ZERO)));
    xvxyz[1] = _mm_max_ps(_mm_set1_ps(0),
                          _mm_min_ps(_mm_set1_ps(0xffff),
                                     _mm_round_ps(xvxyz[1], _MM_FROUND_TO_ZERO)));
    __m128i loadaddrh = _mm_cvttps_epi32(xvxyz[0]);
    __m128i loadaddrv = _mm_cvttps_epi32(xvxyz[1]);
#ifdef __AVX__
    __m256 vlab,
           vxyz = { cielab_cbrt[_mm_extract_epi32(loadaddrh,1)],
                    cielab_cbrt[_mm_extract_epi32(loadaddrh,0)],
                    cielab_cbrt[_mm_extract_epi32(loadaddrh,1)],
                    0,
                    cielab_cbrt[_mm_extract_epi32(loadaddrv,1)],
                    cielab_cbrt[_mm_extract_epi32(loadaddrv,0)],
                    cielab_cbrt[_mm_extract_epi32(loadaddrv,1)],
                    0},
           vxyz2 =  {0,
                     cielab_cbrt[_mm_extract_epi32(loadaddrh,1)],
                     cielab_cbrt[_mm_extract_epi32(loadaddrh,2)],
                     cielab_cbrt[_mm_extract_epi32(loadaddrh,0)],
                     0,
                     cielab_cbrt[_mm_extract_epi32(loadaddrv,1)],
                     cielab_cbrt[_mm_extract_epi32(loadaddrv,2)],
                     cielab_cbrt[_mm_extract_epi32(loadaddrv,0)]};

    vlab = _mm256_sub_ps(vxyz,vxyz2);
    vlab = _mm256_mul_ps(vlab, _mm256_setr_ps(116,500,200,0,116,500,200,0));
    vlab = _mm256_sub_ps(vlab, _mm256_setr_ps(16,0,0,0,16,0,0,0));
    vlab = _mm256_mul_ps(vlab,_mm256_set1_ps(64));
    vlab = _mm256_round_ps(vlab, _MM_FROUND_TO_ZERO);
    __m256i vlabi = _mm256_cvtps_epi32(vlab);
    return _mm_packs_epi32(_mm256_castsi256_si128(vlabi), ((__m128i*)&vlabi)[1]);
#else
    __m128 vlabh, vxyzh = {cielab_cbrt[_mm_extract_epi32(loadaddrh,0)],
                           cielab_cbrt[_mm_extract_epi32(loadaddrh,1)],
                           cielab_cbrt[_mm_extract_epi32(loadaddrh,2)],
                           0};
    __m128 vlabv, vxyzv = {cielab_cbrt[_mm_extract_epi32(loadaddrv,0)],
                           cielab_cbrt[_mm_extract_epi32(loadaddrv,1)],
                           cielab_cbrt[_mm_extract_epi32(loadaddrv,2)],
                           0};

    vlabh = _mm_sub_ps(_mm_shuffle_ps(vxyzh,vxyzh,_MM_SHUFFLE(0,1,0,1)),
                       _mm_shuffle_ps(vxyzh,vxyzh,_MM_SHUFFLE(0,2,1,3)));
    vlabh = _mm_mul_ps(vlabh,_mm_setr_ps(116,500,200,0));
    vlabh = _mm_sub_ps(vlabh,_mm_setr_ps(16,0,0,0));
    vlabh = _mm_mul_ps(vlabh,_mm_set_ps1(64));
    vlabh = _mm_round_ps(vlabh, _MM_FROUND_TO_ZERO);

    vlabv = _mm_sub_ps(_mm_shuffle_ps(vxyzv,vxyzv,_MM_SHUFFLE(0,1,0,1)),
                       _mm_shuffle_ps(vxyzv,vxyzv,_MM_SHUFFLE(0,2,1,3)));
    vlabv = _mm_mul_ps(vlabv,_mm_setr_ps(116,500,200,0));
    vlabv = _mm_sub_ps(vlabv,_mm_setr_ps(16,0,0,0));
    vlabv = _mm_mul_ps(vlabv,_mm_set_ps1(64));
    vlabv = _mm_round_ps(vlabv, _MM_FROUND_TO_ZERO);

    return _mm_set_epi64(_mm_cvtps_pi16(vlabv),_mm_cvtps_pi16(vlabh));
#endif
}

void ahd_interpolate_tile(int top, char * buffer)
{
    int i, j, row, col, tr, tc, c, d, val, hm[2];
    const int dir[4] = { -1, 1, -width, width };
    __m128i ldiff[2], abdiff[2];
    union hvrgbpix (*rgb)[width] = (union hvrgbpix (*)[width])buffer;
    union hvrgbpix *rix;
    union rgbpix * pix;
    union hvrgbpix (*lab)[width];
    short (*lix)[8];
    char (*homo)[TS][width];
    lab  = (union hvrgbpix (*)[width])(buffer + 16*width*TS);
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
            rix = &rgb[row-top][1];

            val = ((pix[-1].g + pix[0].c[c1] + pix[1].g) * 2 - pix[-2].c[c1] - pix[2].c[c1]) >> 2;
            rix[0].h.g = ULIM(val,pix[-1].g,pix[1].g);
            val = ((pix[-width].g + pix[0].c[c1] + pix[width].g) * 2 - pix[-2*width].c[c1] - pix[2*width].c[c1]) >> 2;
            rix[0].v.g = ULIM(val,pix[-width].g,pix[width].g);
            for (col=left+1; col < width-3; col+=2) {
                pix = (union rgbpix*)image + rowx*width+col+1;

                union hvrgbpix rixr, rix0;

                rix = &rgb[rowx-top][col-left]+1;

                signed pix_diag = pix[-width-1].c[c1] + pix[-width+1].c[c1];
                signed pix_ul = pix[-width-1].c[c1];
                rixr.vec = _mm_set1_epi16(pix[-1].g);
                signed pix_lr = pix[-2].c[c2] + pix[0].c[c2];
                rix0.h.c[c2] = rix0.v.c[c2]  = pix[0].c[c2];
                pix_diag += pix[width-1].c[c1] + pix[width+1].c[c1] + 1;
                signed pix_dl = pix[width-1].c[c1];

                //fully loaded
                __m128i rix_dr =               _mm_setr_epi32(pix[width].g,       pix[width-1].c[c1], pix[1].g, pix[-width+1].c[c1]);
                rix_dr = _mm_add_epi32(rix_dr,_mm_setr_epi32(pix[width+1].c[c1],  pix[width+3].c[c1], pix[width+1].c[c1], 0));
                rix_dr = _mm_add_epi32(rix_dr,_mm_setr_epi32(pix[width+2].g,      0,                  pix[2*width+1].g, pix[3*width+1].c[c1]));
                rix_dr = _mm_mullo_epi32(rix_dr,_mm_setr_epi32(2,1,2,1));
                //half loaded
                rix_dr = _mm_hsub_epi32(rix_dr,_mm_setzero_si128());
                rix_dr = _mm_srai_epi32(rix_dr,2);
                __m128i a = _mm_setr_epi32(pix[width].g,pix[1].g,0,0);
                __m128i b = _mm_setr_epi32(pix[width+2].g,pix[2*width+1].g,0,0);
                __m128i m = _mm_min_epi32(a,b);
                __m128i M = _mm_max_epi32(a,b);
                rix_dr = _mm_min_epi32(rix_dr,M);
                rix_dr = _mm_max_epi32(rix_dr,m);

                signed pix_udr = pix_ul + pix_dl;

                signed rix0_ul = rix[-width-1].h.g;
                signed rix1_ul = rix[-width-1].v.g;
                __m128i rix_ur = _mm_setr_epi32(rix[-width+1].h.g, rix[-width+1].v.g, 0, 0);
                signed rix0_rr = rix[-2].h.g;
                signed rix1_rr = rix[-2].v.g;

                rix0.h.g = rix[0].h.g;
                rix0.v.g = rix[0].v.g;
                signed rix0_dl = rix[width-1].h.g;
                signed rix1_dl = rix[width-1].v.g;

                // fully loaded
                __m128i rix_udr = _mm_setr_epi32(rix0_ul, rix1_ul, rix0_rr, rix1_rr);
                rix_udr = _mm_add_epi32(rix_udr, _mm_setr_epi32(rix0_dl, rix1_dl, rix0.h.g, rix0.v.g));
                __m128i v2 = _mm_set_epi32(pix_lr, pix_lr, pix_udr, pix_udr);
                v2 = _mm_sub_epi32(v2, rix_udr);
                v2 = _mm_srai_epi32(v2,1);
                v2 = _mm_add_epi32(v2,_mm_cvtepu16_epi32(rixr.vec));
                v2 = _mm_max_epi32(v2, _mm_setzero_si128());
                v2 = _mm_min_epi32(v2, _mm_set1_epi32(0xffff));
                rixr.h.c[c2] = _mm_extract_epi32(v2,2);
                rixr.v.c[c2] = _mm_extract_epi32(v2,3);
                rixr.h.c[c1] = _mm_extract_epi32(v2,0);
                rixr.v.c[c1] = _mm_extract_epi32(v2,1);

                // following only uses 64 bit
                __m128i v1 = _mm_set1_epi32(pix_diag);
                v1 = _mm_sub_epi32(v1, rix_ur);
                v1 = _mm_sub_epi32(v1, rix_dr);
                v1 = _mm_sub_epi32(v1, rix_udr);
                v1 = _mm_srai_epi32(v1,2);
                v1 = _mm_add_epi32(v1, _mm_setr_epi32(rix0.h.g, rix0.v.g, 0, 0));
                v1 = _mm_max_epi32(v1, _mm_setzero_si128());
                v1 = _mm_min_epi32(v1, _mm_set1_epi32(0xffff));
                rix0.h.c[c1] = _mm_extract_epi32(v1,0);
                rix0.v.c[c1] = _mm_extract_epi32(v1,1);


                lab[rowx-top][col-left].vec = cielabv(rixr);
                lab[rowx-top][col-left+1].vec = cielabv(rix0);

                _mm_store_si128(&rix[-1].vec,rixr.vec);
                _mm_store_si128(&rix[0].vec,rix0.vec);

                rix[width+1].h.g = _mm_extract_epi32(rix_dr,0);
                rix[width+1].v.g = _mm_extract_epi32(rix_dr,1);
            }
        } else {
            int c1 = FC(rowx+1,left+2),
                c2 = FC(rowx,left+1);

            pix = (union rgbpix*)image + row*width+left;
            rix = &rgb[row-top][0];
            val = ((pix[-1].g + pix[0].c[c1] + pix[1].g) * 2 - pix[-2].c[c1] - pix[2].c[c1]) >> 2;
            rix[0].h.g = ULIM(val,pix[-1].g,pix[1].g);
            val = ((pix[-width].g + pix[0].c[c1] + pix[width].g) * 2 - pix[-2*width].c[c1] - pix[2*width].c[c1]) >> 2;
            rix[0].v.g = ULIM(val,pix[-width].g,pix[width].g);
            for (col=left+1; col < width-3; col+=2) {
                pix = (union rgbpix*)image + rowx*width+col;

                union hvrgbpix rix0, rixr;

                rix = &rgb[rowx-top][col-left];

                signed pix_diag = pix[-width-1].c[c1] + pix[-width+1].c[c1];
                signed pix_ur = pix[-width+1].c[c1];
                rix0.h.c[c2] = rix0.v.c[c2] = pix[0].c[c2];
                signed pix_lr = pix[0].c[c2] + pix[2].c[c2];
                rixr.h.g = rixr.v.g = pix[1].g;
                pix_diag += pix[width-1].c[c1] + pix[width+1].c[c1]+1;
                signed pix_dr = pix[width+1].c[c1];

                __m128i rix_dr =                _mm_setr_epi32(pix[width].g,        pix[width-1].c[c1], rixr.v.g, pix_ur             );
                rix_dr = _mm_add_epi32(rix_dr,  _mm_setr_epi32(pix[width+1].c[c1],  pix[width+3].c[c1], pix_dr, 0));
                rix_dr = _mm_add_epi32(rix_dr,  _mm_setr_epi32(pix[width+2].g,      0,                  pix[2*width+1].g, pix[3*width+1].c[c1]));
                rix_dr = _mm_mullo_epi32(rix_dr,_mm_setr_epi32(2,1,2,1));
                rix_dr = _mm_hsub_epi32(rix_dr,_mm_setzero_si128());
                rix_dr = _mm_srai_epi32(rix_dr,2);
                __m128i a = _mm_setr_epi32(pix[width].g,pix[1].g,0,0);
                __m128i b = _mm_setr_epi32(pix[width+2].g,pix[2*width+1].g,0,0);
                __m128i m = _mm_min_epi32(a,b);
                __m128i M = _mm_max_epi32(a,b);
                rix_dr = _mm_min_epi32(rix_dr,M);
                rix_dr = _mm_max_epi32(rix_dr,m);

                signed pix_udr = pix_ur+pix_dr;

                __m128i rix_ul = _mm_setr_epi32(rix[-width-1].h.g, rix[-width-1].v.g, 0, 0);
                __m128i rix_ur = _mm_setr_epi32(rix[-width+1].h.g, rix[-width+1].v.g, rix[2].h.g, rix[2].v.g);
                __m128i rix0g = _mm_setr_epi32(rix[0].h.g, rix[0].v.g,0,0);
                rix0.h.g = rix[0].h.g;
                rix0.v.g = rix[0].v.g;
                __m128i rix_dl = _mm_setr_epi32(rix[width-1].h.g, rix[width-1].v.g,0,0);

                rix_dr = _mm_unpacklo_epi64(rix_dr, rix0g);
                __m128i rix_udr = _mm_add_epi32(rix_ur, rix_dr);

                __m128i v2 = _mm_setr_epi32(pix_udr, pix_udr, pix_lr, pix_lr);
                v2 = _mm_sub_epi32(v2, rix_udr);
                v2 = _mm_srai_epi32(v2, 1);
                v2 = _mm_add_epi32(v2, _mm_set1_epi32(rixr.h.g));
                v2 = _mm_max_epi32(v2, _mm_setzero_si128());
                v2 = _mm_min_epi32(v2, _mm_set1_epi32(0xffff));

                rixr.h.c[c2] = _mm_extract_epi32(v2,2);
                rixr.v.c[c2] = _mm_extract_epi32(v2,3);
                rixr.h.c[c1] = _mm_extract_epi32(v2,0);
                rixr.v.c[c1] = _mm_extract_epi32(v2,1);

                __m128i v1 = _mm_set1_epi32(pix_diag);
                v1 = _mm_sub_epi32(v1, rix_ul);
                v1 = _mm_sub_epi32(v1, rix_dl);
                v1 = _mm_sub_epi32(v1, rix_udr);
                v1 = _mm_srai_epi32(v1,2);
                v1 = _mm_add_epi32(v1, rix0g);
                v1 = _mm_max_epi32(v1, _mm_setzero_si128());
                v1 = _mm_min_epi32(v1, _mm_set1_epi32(0xffff));
                rix0.h.c[c1] = _mm_extract_epi32(v1,0);
                rix0.v.c[c1] = _mm_extract_epi32(v1,1);

                lab[rowx-top][col-left].vec = cielabv(rix0);
                lab[rowx-top][col-left+1].vec = cielabv(rixr);

                _mm_store_si128(&rix[0].vec,rix0.vec);
                _mm_store_si128(&rix[1].vec,rixr.vec);

                rix[width+1].h.g = _mm_extract_epi32(rix_dr,0);
                rix[width+1].v.g = _mm_extract_epi32(rix_dr,1);
            }
        }
    }
/*  Build homogeneity maps from the CIELab images:    */
    memset (homo, 0, 2*width*TS);
    for (row=top+2; row < top+TS-2 && row < height-4; row++) {
        tr = row-top;
        for (col=left+2; col < width-4; col++) {
            tc = col-left;
            lix = (short (*)[8])lab[tr][tc].h.c;
            __m128i lixd0, lix0 =  _mm_lddqu_si128((__m128i*)&(lix[0][0]));
            __m128i lixdi01,lixi01 = _mm_lddqu_si128((__m128i*)&(lix[dir[0]][0]));
            __m128i t1 =     _mm_lddqu_si128((__m128i*)&(lix[dir[1]][0]));
            __m128i lixdi23,lixi23 = _mm_lddqu_si128((__m128i*)&(lix[dir[2]][0]));
            __m128i t2 =     _mm_lddqu_si128((__m128i*)&(lix[dir[3]][0]));
            lixdi01 = _mm_unpackhi_epi64(lixi01,t1);
            lixdi23 = _mm_unpackhi_epi64(lixi23,t2);
            lixd0 = _mm_unpackhi_epi64(lix0, lix0);
            lixi01 = _mm_unpacklo_epi64(lixi01,t1);
            lixi23 = _mm_unpacklo_epi64(lixi23,t2);
            lix0 = _mm_unpacklo_epi64(lix0, lix0);

            lixi01 = _mm_sub_epi16(lix0, lixi01);
            lixi01 = _mm_abs_epi16(lixi01);
            lixi23 = _mm_sub_epi16(lix0, lixi23);
            lixi23 = _mm_abs_epi16(lixi23);

            lixdi01 = _mm_sub_epi16(lixd0, lixdi01);
            lixdi01 = _mm_abs_epi16(lixdi01);
            lixdi23 = _mm_sub_epi16(lixd0, lixdi23);
            lixdi23 = _mm_abs_epi16(lixdi23);

            ldiff[0] = _mm_unpacklo_epi64(_mm_shuffle_epi32(lixi01, _MM_SHUFFLE(2,0,2,0)), _mm_shuffle_epi32(lixi23, _MM_SHUFFLE(2,0,2,0)));
            ldiff[0] = _mm_and_si128(ldiff[0], _mm_setr_epi16(0xffff,0,0xffff,0,0xffff,0,0xffff,0));
            ldiff[1] = _mm_unpacklo_epi64(_mm_shuffle_epi32(lixdi01, _MM_SHUFFLE(2,0,2,0)), _mm_shuffle_epi32(lixdi23, _MM_SHUFFLE(2,0,2,0)));
            ldiff[1] = _mm_and_si128(ldiff[1], _mm_setr_epi16(0xffff,0,0xffff,0,0xffff,0,0xffff,0));

            lixi01 = _mm_srli_epi64(lixi01,16);
            lixi01 = _mm_shuffle_epi32(lixi01, _MM_SHUFFLE(0,2,2,0));
            lixi01 = _mm_cvtepu16_epi32(lixi01);
            lixi01 = _mm_mullo_epi32(lixi01,lixi01);

            lixi23 = _mm_srli_epi64(lixi23,16);
            lixi23 = _mm_shuffle_epi32(lixi23, _MM_SHUFFLE(0,2,2,0));
            lixi23 = _mm_cvtepu16_epi32(lixi23);
            lixi23 = _mm_mullo_epi32(lixi23,lixi23);
            abdiff[0] = _mm_hadd_epi32(lixi01,lixi23);

            lixdi01 = _mm_srli_epi64(lixdi01,16);
            lixdi01 = _mm_shuffle_epi32(lixdi01, _MM_SHUFFLE(0,2,2,0));
            lixdi01 = _mm_cvtepu16_epi32(lixdi01);
            lixdi01 = _mm_mullo_epi32(lixdi01,lixdi01);

            lixdi23 = _mm_srli_epi64(lixdi23,16);
            lixdi23 = _mm_shuffle_epi32(lixdi23, _MM_SHUFFLE(0,2,2,0));
            lixdi23 = _mm_cvtepu16_epi32(lixdi23);
            lixdi23 = _mm_mullo_epi32(lixdi23,lixdi23);
            abdiff[1] = _mm_hadd_epi32(lixdi01,lixdi23);


            __m128i t3=_mm_unpacklo_epi32(ldiff[0], abdiff[0]);
            t2=_mm_unpackhi_epi32(ldiff[1], abdiff[1]);
            t1=_mm_unpacklo_epi64(t3,t2);
            t2=_mm_unpackhi_epi64(t3,t2);
            t3 = _mm_max_epi32(t1,t2); //Ml0, Ma0, Ml1, Ma1
            t3 = _mm_min_epi32(t3, _mm_shuffle_epi32(t3, _MM_SHUFFLE(0,0,3,2)));
            t1 = _mm_shuffle_epi32(t3, _MM_SHUFFLE(0,0,0,0)); //leps, leps, leps, leps
            t2 = _mm_shuffle_epi32(t3, _MM_SHUFFLE(1,1,1,1)); //abeps,abeps,abeps,abeps

            t3 = _mm_or_si128(_mm_cmpgt_epi32(ldiff[0], t1), _mm_cmpgt_epi32(abdiff[0], t2));
            t3 = _mm_add_epi32(t3, _mm_set_epi32(1,1,1,1));

            t2 = _mm_or_si128(_mm_cmpgt_epi32(ldiff[1], t1), _mm_cmpgt_epi32(abdiff[1], t2));
            t2 = _mm_add_epi32(t2, _mm_set_epi32(1,1,1,1));

            t3 = _mm_hadd_epi32(t3,t3);
            t3 = _mm_hadd_epi32(t3,t3);

            t2 = _mm_hadd_epi32(t2,t2);
            t2 = _mm_hadd_epi32(t2,t2);

            homo[0][tr][tc]+=_mm_extract_epi32(t3,0);
            homo[1][tr][tc]+=_mm_extract_epi32(t2,0);
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
