/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
 */

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
 */

#pragma once
#include "simd_utils_constants.h"
#include <immintrin.h>

static inline v4sf log_ps(v4sf x)
{
    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;

    v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

    x = _mm_max_ps(x, *(v4sf *) _ps_min_norm_pos); /* cut off denormalized stuff */

    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    /* keep only the fractional part */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_mant_mask);
    x = _mm_or_ps(x, *(v4sf *) _ps_0p5);

    emm0 = _mm_sub_epi32(emm0, *(v4si *) _pi32_0x7f);
    v4sf e = _mm_cvtepi32_ps(emm0);

    e = _mm_add_ps(e, one);

    /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
     */
    v4sf mask = _mm_cmplt_ps(x, *(v4sf *) _ps_cephes_SQRTHF);
    v4sf tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);


    v4sf z = _mm_mul_ps(x, x);

    v4sf y = _mm_fmadd_ps(*(v4sf *) _ps_cephes_log_p0, x, *(v4sf *) _ps_cephes_log_p1);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p2);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p3);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p4);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p5);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p6);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p7);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_log_p8);
    y = _mm_mul_ps(y, x);

    y = _mm_mul_ps(y, z);

    y = _mm_fmadd_ps(e, *(v4sf *) _ps_cephes_log_q1, y);
    y = _mm_fnmadd_ps(z, *(v4sf *) _ps_0p5, y);

    tmp = _mm_fmadd_ps(e, *(v4sf *) _ps_cephes_log_q2, y);
    x = _mm_add_ps(x, tmp);
    x = _mm_or_ps(x, invalid_mask);  // negative arg will be NAN
    return x;
}

static inline v4sf exp_ps(v4sf x)
{
    v4sf fx;
    v4si emm0;
    v4sf one = *(v4sf *) _ps_1;

    x = _mm_min_ps(x, *(v4sf *) _ps_exp_hi);
    x = _mm_max_ps(x, *(v4sf *) _ps_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm_fmadd_ps(x, *(v4sf *) _ps_cephes_LOG2EF, *(v4sf *) _ps_0p5);
    fx = _mm_round_ps(fx, _MM_FROUND_FLOOR);

    x = _mm_fnmadd_ps(fx, *(v4sf *) _ps_cephes_exp_C1, x);
    x = _mm_fnmadd_ps(fx, *(v4sf *) _ps_cephes_exp_C2, x);

    v4sf z = _mm_mul_ps(x, x);

    v4sf y = _mm_fmadd_ps(*(v4sf *) _ps_cephes_exp_p0, x, *(v4sf *) _ps_cephes_exp_p1);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_exp_p2);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_exp_p3);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_exp_p4);
    y = _mm_fmadd_ps(y, x, *(v4sf *) _ps_cephes_exp_p5);
    y = _mm_fmadd_ps(y, z, x);
    y = _mm_add_ps(y, one);

    /* build 2^n */
    emm0 = _mm_cvttps_epi32(fx);
    emm0 = _mm_add_epi32(emm0, *(v4si *) _pi32_0x7f);
    emm0 = _mm_slli_epi32(emm0, 23);
    v4sf pow2n = _mm_castsi128_ps(emm0);

    y = _mm_mul_ps(y, pow2n);
    return y;
}

static inline v4sf sin_ps(v4sf x)
{  // any x
    v4sf sign_bit, y;

    v4si emm0, emm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm_and_ps(sign_bit, *(v4sf *) _ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, *(v4sf *) _ps_cephes_FOPI);

    /* store the integer part of y in mm0 */
    emm2 = _mm_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, *(v4si *) _pi32_1);
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    /* get the swap sign flag */
    emm0 = _mm_and_si128(emm2, *(v4si *) _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
     */
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

    v4sf swap_sign_bit = _mm_castsi128_ps(emm0);
    v4sf poly_mask = _mm_castsi128_ps(emm2);
    sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP1, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP2, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4sf z = _mm_mul_ps(x, x);

    y = _mm_fmadd_ps(*(v4sf *) _ps_coscof_p0, z, *(v4sf *) _ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(v4sf *) _ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(v4sf *) _ps_0p5, y);
    y = _mm_add_ps(y, *(v4sf *) _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4sf y2 = _mm_fmadd_ps(*(v4sf *) _ps_sincof_p0, z, *(v4sf *) _ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(v4sf *) _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm_and_ps(poly_mask, y2);
    y = _mm_andnot_ps(poly_mask, y);
    y = _mm_add_ps(y, y2);
#endif

    /* update the sign */
    y = _mm_xor_ps(y, sign_bit);
    return y;
}

/* almost the same as sin_ps */
static inline v4sf cos_ps(v4sf x)
{  // any x
    v4sf y;

    v4si emm0, emm2;

    /* take the absolute value */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, *(v4sf *) _ps_cephes_FOPI);

    /* store the integer part of y in mm0 */
    emm2 = _mm_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, *(v4si *) _pi32_1);
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    emm2 = _mm_sub_epi32(emm2, *(v4si *) _pi32_2);

    /* get the swap sign flag */
    emm0 = _mm_andnot_si128(emm2, *(v4si *) _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    /* get the polynom selection mask */
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

    v4sf sign_bit = _mm_castsi128_ps(emm0);
    v4sf poly_mask = _mm_castsi128_ps(emm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP1, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP2, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4sf z = _mm_mul_ps(x, x);

    y = _mm_fmadd_ps(*(v4sf *) _ps_coscof_p0, z, *(v4sf *) _ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(v4sf *) _ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(v4sf *) _ps_0p5, y);
    y = _mm_add_ps(y, *(v4sf *) _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4sf y2 = _mm_fmadd_ps(*(v4sf *) _ps_sincof_p0, z, *(v4sf *) _ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(v4sf *) _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm_blendv_ps(y, y2, poly_mask);
#else
    y2 = _mm_and_ps(poly_mask, y2);
    y = _mm_andnot_ps(poly_mask, y);
    y = _mm_add_ps(y, y2);
#endif

    /* update the sign */
    y = _mm_xor_ps(y, sign_bit);

    return y;
}


/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static inline void sincos_ps(v4sf x, v4sf *s, v4sf *c)
{
    v4sf xmm1, xmm2, sign_bit_sin, y;

    v4si emm0, emm2, emm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm_and_ps(x, *(v4sf *) _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm_and_ps(sign_bit_sin, *(v4sf *) _ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, *(v4sf *) _ps_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = _mm_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, *(v4si *) _pi32_1);
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm_and_si128(emm2, *(v4si *) _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    v4sf swap_sign_bit_sin = _mm_castsi128_ps(emm0);

    /* get the polynom selection mask for the sine*/
    emm2 = _mm_and_si128(emm2, *(v4si *) _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    v4sf poly_mask = _mm_castsi128_ps(emm2);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP1, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP2, x);
    x = _mm_fmadd_ps(y, *(v4sf *) _ps_minus_cephes_DP3, x);

    emm4 = _mm_sub_epi32(emm4, *(v4si *) _pi32_2);
    emm4 = _mm_andnot_si128(emm4, *(v4si *) _pi32_4);
    emm4 = _mm_slli_epi32(emm4, 29);
    v4sf sign_bit_cos = _mm_castsi128_ps(emm4);

    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);


    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v4sf z = _mm_mul_ps(x, x);

    y = _mm_fmadd_ps(*(v4sf *) _ps_coscof_p0, z, *(v4sf *) _ps_coscof_p1);
    y = _mm_fmadd_ps(y, z, *(v4sf *) _ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    y = _mm_fnmadd_ps(z, *(v4sf *) _ps_0p5, y);
    y = _mm_add_ps(y, *(v4sf *) _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v4sf y2 = _mm_fmadd_ps(*(v4sf *) _ps_sincof_p0, z, *(v4sf *) _ps_sincof_p1);
    y2 = _mm_fmadd_ps(y2, z, *(v4sf *) _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm_blendv_ps(y, y2, poly_mask);
    xmm2 = _mm_blendv_ps(y2, y, poly_mask);
#else
    v4sf ysin2 = _mm_and_ps(poly_mask, y2);
    v4sf ysin1 = _mm_andnot_ps(poly_mask, y);
    y2 = _mm_sub_ps(y2, ysin2);
    y = _mm_sub_ps(y, ysin1);
    xmm1 = _mm_add_ps(ysin1, ysin2);
    xmm2 = _mm_add_ps(y, y2);
#endif

    /* update the sign */
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}