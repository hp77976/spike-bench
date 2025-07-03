#include <immintrin.h>

/* __m128 is ugly to write */
typedef __m256  v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int   (avx)
typedef __m128i v4si; // vector of 8 int   (avx)

typedef __m128 v4sf;  // vector of 4 float (sse1)
typedef __m64  v2si;  // vector of 2 int (mmx)

typedef __m512  v16sf; // vector of 16 float (avx512)
typedef __m512i v16si; // vector of 16 int   (avx512)
typedef __m512i v8sid; // vector of 8 64bits int   (avx512)
typedef __m256i v8si;  // vector of 8 int   (avx)
typedef __m512d v8sd;  // vector of 8 double (avx512)