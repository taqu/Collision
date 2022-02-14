#pragma once
#ifndef INC_BVH_BINQBVH_H__
#    define INC_BVH_BINQBVH_H__
/**
  @file BinQBVH.h
  @author t-sakai
  @date 2018/01/22 create
  */
#    if defined(BVH_UE)
#        include <CoreMinimal.h>
using bvh_int32_t = int32;
using bvh_uint32_t = uint32;

#    else
#        include <cassert>
#        include <cstdint>
#        include <cstring>
#        if defined(__GNUC__) || defined(__clang__)
#            include <cstdlib>
#        endif
#        include <limits>
#        include <utility>

using bvh_int32_t = int32_t;
using bvh_uint32_t = uint32_t;
#    endif

#    if defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX2__)
//#        pragma message("use x86 simd")
#        define BVH_SSE (1)
#        include <xmmintrin.h>
using vector4_t = __m128;
using uvector4_t = __m128;

inline vector4_t load(const float* x)
{
    return _mm_loadu_ps(x);
}

inline void store(float* x0, vector4_t x1)
{
    _mm_storeu_ps(x0, x1);
}

inline vector4_t setzero()
{
    return _mm_setzero_ps();
}

inline vector4_t set1(float x)
{
    return _mm_set1_ps(x);
}

inline uvector4_t set1_u32(float x)
{
    return _mm_set1_ps(x);
}

inline vector4_t maximum4(vector4_t x0, vector4_t x1)
{
    return _mm_max_ps(x0, x1);
}

inline vector4_t minimum4(vector4_t x0, vector4_t x1)
{
    return _mm_min_ps(x0, x1);
}

inline vector4_t add(vector4_t x0, vector4_t x1)
{
    return _mm_add_ps(x0, x1);
}

inline vector4_t sub(vector4_t x0, vector4_t x1)
{
    return _mm_sub_ps(x0, x1);
}

inline vector4_t mul(vector4_t x0, vector4_t x1)
{
    return _mm_mul_ps(x0, x1);
}

inline vector4_t cmple(vector4_t x0, vector4_t x1)
{
    return _mm_cmple_ps(x0, x1);
}

inline vector4_t cmpge(vector4_t x0, vector4_t x1)
{
    return _mm_cmpge_ps(x0, x1);
}

inline bvh_int32_t movemask(vector4_t x)
{
    return _mm_movemask_ps(x);
}

inline vector4_t and4(vector4_t x0, vector4_t x1)
{
    return _mm_and_ps(x0, x1);
}

#    endif

#if defined(__ARM_NEON)
//#        pragma message("use arm neon")
#        define BVH_NEON (1)
#        include <arm_neon.h>
using vector4_t = float32x4_t;
using uvector4_t = uint32x4_t;

inline vector4_t load(const float* x)
{
    return vld1q_f32(x);
}

inline void store(float* x0, vector4_t x1)
{
    vst1q_f32(x0, x1);
}

inline vector4_t setzero()
{
    return vdupq_n_f32(0.0f);
}

inline vector4_t set1(float x)
{
    return vdupq_n_f32(x);
}

inline uvector4_t set1_u32(bvh_uint32_t x)
{
    return vdupq_n_u32(x);
}

inline vector4_t maximum4(vector4_t x0, vector4_t x1)
{
    return vmaxq_f32(x0, x1);
}

inline vector4_t minimum4(vector4_t x0, vector4_t x1)
{
    return vminq_f32(x0, x1);
}

inline vector4_t add(vector4_t x0, vector4_t x1)
{
    return vaddq_f32(x0, x1);
}

inline vector4_t sub(vector4_t x0, vector4_t x1)
{
    return vsubq_f32(x0, x1);
}

inline vector4_t mul(vector4_t x0, vector4_t x1)
{
    return vmulq_f32(x0, x1);
}

inline uvector4_t cmple(vector4_t x0, vector4_t x1)
{
    return vcleq_f32(x0, x1);
}

inline uvector4_t cmpge(vector4_t x0, vector4_t x1)
{
    return vcgeq_f32(x0, x1);
}

bvh_int32_t movemask(uvector4_t x);

inline uvector4_t and4(uvector4_t x0, uvector4_t x1)
{
    return vandq_u32(x0, x1);
}
#    endif

#    ifdef __ARM_FEATURE_SVE
#        pragma message("use arm sve")
#        include <arm_sve.h>
#    endif

#    if !defined(BVH_SSE) && !defined(BVH_NEON)
#        define BVH_SOFT (1)

struct vector4_t
{
    float x_;
    float y_;
    float z_;
    float w_;
};

struct uvector4_t
{
    bvh_uint32_t x_;
    bvh_uint32_t y_;
    bvh_uint32_t z_;
    bvh_uint32_t w_;
};

inline vector4_t load(const float* x)
{
    return {x[0], x[1], x[2], x[3]};
}

inline void store(float* x0, vector4_t x1)
{
    x0[0] = x1.x_;
    x0[1] = x1.y_;
    x0[2] = x1.z_;
    x0[3] = x1.w_;
}

inline vector4_t setzero()
{
    return {};
}

inline vector4_t set1(float x)
{
    return {x, x, x, x};
}

inline uvector4_t set1_u32(float x)
{
    union Union
    {
        float f_;
        bvh_uint32_t u_;
    };
    Union t;
    t.f_ = x;
    return {t.u_, t.u_, t.u_, t.u_};
}

inline vector4_t maximum4(vector4_t x0, vector4_t x1)
{
    auto maximum_ = [](float x0, float x1) {
        return x0 < x1 ? x1 : x0;
    };
    return {maximum_(x0.x_, x1.x_), maximum_(x0.y_, x1.y_), maximum_(x0.z_, x1.z_), maximum_(x0.w_, x1.w_)};
}

inline vector4_t minimum4(vector4_t x0, vector4_t x1)
{
    auto minimum_ = [](float x0, float x1) {
        return x0 < x1 ? x0 : x1;
    };
    return {minimum_(x0.x_, x1.x_), minimum_(x0.y_, x1.y_), minimum_(x0.z_, x1.z_), minimum_(x0.w_, x1.w_)};
}

inline vector4_t add(vector4_t x0, vector4_t x1)
{
    return {x0.x_ + x1.x_, x0.y_ + x1.y_, x0.z_ + x1.z_, x0.w_ + x1.w_};
}

inline vector4_t sub(vector4_t x0, vector4_t x1)
{
    return {x0.x_ - x1.x_, x0.y_ - x1.y_, x0.z_ - x1.z_, x0.w_ - x1.w_};
}

inline vector4_t mul(vector4_t x0, vector4_t x1)
{
    return {x0.x_ * x1.x_, x0.y_ * x1.y_, x0.z_ * x1.z_, x0.w_ * x1.w_};
}

inline uvector4_t cmple(vector4_t x0, vector4_t x1)
{
    auto cmple_ = [](float x0, float x1) {
        return x0 <= x1 ? 0xFFFFFFFFU : 0x00000000U;
    };
    return {cmple_(x0.x_, x1.x_), cmple_(x0.y_, x1.y_), cmple_(x0.z_, x1.z_), cmple_(x0.w_, x1.w_)};
}

inline uvector4_t cmpge(vector4_t x0, vector4_t x1)
{
    auto cmpge_ = [](float x0, float x1) {
        return x0 >= x1 ? 0xFFFFFFFFU : 0x00000000U;
    };
    return {cmpge_(x0.x_, x1.x_), cmpge_(x0.y_, x1.y_), cmpge_(x0.z_, x1.z_), cmpge_(x0.w_, x1.w_)};
}

inline bvh_int32_t movemask(uvector4_t x)
{
    bvh_uint32_t mask[4] = {x.x_, x.y_, x.z_, x.w_};
    return ((mask[3] & 0x01U) << 3) | ((mask[2] & 0x01U) << 2) | ((mask[1] & 0x01U) << 1) | ((mask[0] & 0x01U) << 0);
}

inline uvector4_t and4(uvector4_t x0, uvector4_t x1)
{
    return {x0.x_ & x1.x_, x0.y_ & x1.y_, x0.z_ & x1.z_, x0.w_ & x1.w_};
}
#    endif

#if defined(BVH_UE)
static constexpr float bvh_limits_max = (TNumericLimits<float>::Max)();
#    else
static constexpr float bvh_limits_max = (std::numeric_limits<float>::max)();
#    endif

#    ifdef __cplusplus
#        if 201103L <= __cplusplus || 1900 <= _MSC_VER
#            define BVH_CPP11 1
#        endif
#    endif

#    ifndef BVH_NULL
#        ifdef __cplusplus
#            ifdef BVH_CPP11
#                define BVH_NULL nullptr
#            endif
#        else
#            define BVH_NULL ((void*)0)
#        endif
#    endif

#    ifdef _MSC_VER
#        define BVH_ALIGN16 __declspec(align(16))
#        define BVH_ALIGN(x) __declspec(align(x))
#    else
#        define BVH_ALIGN16 __attribute__((aligned(16)))
#        define BVH_ALIGN(x) __attribute__((aligned(x)))
#    endif

#    define BVH_PLACEMENT_NEW(ptr) new(ptr)
#    define BVH_DELETE(p) \
        delete p; \
        (p) = BVH_NULL
#    define BVH_DELETE_NONULL(p) delete p

#    define BVH_DELETE_ARRAY(p) \
        delete[](p); \
        (p) = BVH_NULL

#    if defined(BVH_UE)
#        define BVH_MALLOC(size) (FMemory::Malloc(size))
#        define BVH_FREE(mem) \
            FMemory::Free(mem); \
            (mem) = BVH_NULL

#        define BVH_ALIGNED_MALLOC(size) (FMemory::Malloc(size, 16U))
#        define BVH_ALIGNED_FREE(mem) \
            FMemory::Free(mem); \
            (mem) = BVH_NULL

#        define BVH_MEMCPY(dst, src, size) FMemory::Memcpy((dst), (src), (size))
#        define BVH_SQRT(x) FGenericPlatformMath::Sqrt((x))

#    else
#        define BVH_MALLOC(size) (::malloc(size))
#        define BVH_FREE(mem) \
            ::free(mem); \
            (mem) = BVH_NULL

#        define BVH_MEMCPY(dst, src, size) ::memcpy((dst), (src), (size))
#        define BVH_SQRT(x) ::sqrtf(x)

#        if defined(_MSC_VER)
#            define BVH_ALIGNED_MALLOC(size) (_aligned_malloc(size, 16U))
#            define BVH_ALIGNED_FREE(mem) \
                _aligned_free(mem); \
                (mem) = BVH_NULL

#        else
#            if 200112 <= _POSIX_C_SOURCE
inline void* BVH_ALIGNED_MALLOC(size_t size)
{
    void* ptr;
    return 0 == posix_memalign(&ptr, 16U, size) ? ptr : BVH_NULL;
}
#            elif defined(_ISOC11_SOURCE)
inline void* BVH_ALIGNED_MALLOC(size_t size)
{
    size = (size + 15ULL) & ~15ULL;
    return aligned_alloc(16ULL, size);
}
#            endif

#            define BVH_ALIGNED_FREE(mem) \
                ::free(mem); \
                (mem) = BVH_NULL
#        endif
#    endif

// Assertion
//-------------------
#    if defined(BVH_UE)
#        define BVH_ASSERT(expression) check(expression)
#    else
#        if defined(_DEBUG)
#            if defined(ANDROID)
#                define BVH_ASSERT(expression) \
                    { \
                        if((expression) == false) { \
                            __android_log_assert("assert", "lime", "%s (%d)", __FILE__, __LINE__); \
                        } \
                    } \
                    while(0)

#            elif defined(__GNUC__)
#                define BVH_ASSERT(expression) (assert(expression))

#            else
#                define BVH_ASSERT(expression) (assert(expression))
#            endif
#        else
#            define BVH_ASSERT(expression)
#        endif
#    endif

namespace bvh
{
#    if defined(BVH_UE)
using s8 = int8;
using s16 = int16;
using s32 = int32;

using u8 = uint8;
using u16 = uint16;
using u32 = uint32;

using f32 = float;
using f64 = double;

#    else
using s8 = int8_t;
using s16 = int16_t;
using s32 = bvh_int32_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = bvh_uint32_t;

using f32 = float;
using f64 = double;
#    endif

static constexpr f32 F32_EPSILON = 1.0e-06F;
static constexpr f32 F32_HITEPSILON = 1.0e-5f;

static constexpr f64 BVH_E = 2.71828182845904523536;
static constexpr f64 BVH_LOG2E = 1.44269504088896340736;
static constexpr f64 BVH_LOG10E = 0.434294481903251827651;
static constexpr f64 BVH_LN2 = 0.693147180559945309417;
static constexpr f64 BVH_LN10 = 2.30258509299404568402;
static constexpr f64 BVH_PI = 3.14159265358979323846;
static constexpr f64 BVH_PI_2 = 1.57079632679489661923;
static constexpr f64 BVH_PI_4 = 0.785398163397448309616;
static constexpr f64 BVH_1_PI = 0.318309886183790671538;
static constexpr f64 BVH_2_PI = 0.636619772367581343076;
static constexpr f64 BVH_2_SQRTPI = 1.12837916709551257390;
static constexpr f64 BVH_SQRT2 = 1.41421356237309504880;
static constexpr f64 BVH_SQRT1_2 = 0.707106781186547524401;

enum class Result : u8
{
    Result_Fail = 0,
    Result_Success = (0x01U << 0),
    Result_Front = (0x01U << 0) | (0x01U << 1),
    Result_Back = (0x01U << 0) | (0x01U << 2),
};

template<class T>
inline T absolute(const T x)
{
#    if defined(BVH_UE)
    return FGenericPlatformMath::Abs(x);
#    else
    return std::abs(x);
#    endif
}

template<class T>
void swap(T& x0, T& x1)
{
    T tmp = x0;
    x0 = x1;
    x1 = tmp;
}

template<class T>
T minimum(const T& x0, const T& x1)
{
    return (x0 < x1) ? x0 : x1;
}

template<class T>
T maximum(const T& x0, const T& x1)
{
    return (x0 < x1) ? x1 : x0;
}

inline bool isEqual(f32 x0, f32 x1)
{
    return (absolute(x0 - x1) < F32_EPSILON);
}

inline bool isZero(f32 x)
{
    return (absolute(x) < F32_EPSILON);
}

inline bool isZeroPositive(f32 x)
{
    return (x < F32_EPSILON);
}

inline bool isZeroNegative(f32 x)
{
    return (-F32_EPSILON < x);
}

template<class T>
T clamp(T val, T low, T high)
{
    if(val <= low) {
        return low;
    } else if(high <= val) {
        return high;
    } else {
        return val;
    }
}

f32 clamp01(f32 v);

// Returned value is undefined, if x==0
u32 leadingzero(u32 x);

template<class T>
struct SortCompFuncType
{
    typedef bool (*SortCmp)(const T& lhs, const T& rhs);
};

/**
  @return true if lhs<rhs, false if lhs>=rhs
  */
// template<class T>
// typedef bool(*SortCompFunc)(const T& lhs, const T& rhs);

template<class T>
bool less(const T& lhs, const T& rhs)
{
    return (lhs < rhs);
}

//--- insertionsort
//------------------------------------------------
/**
 */
template<class T, class U>
void insertionsort(u32 n, T* v, U func)
{
    for(u32 i = 1; i < n; ++i) {
        for(u32 j = i; 0 < j; --j) {
            u32 k = j - 1;
            if(func(v[j], v[k])) {
                bvh::swap(v[j], v[k]);
            } else {
                break;
            }
        }
    }
}

template<class T>
void insertionsort(u32 n, T* v)
{
    insertionsort(n, v, less<T>);
}

//--- heapsort
//------------------------------------------------
/**
 */
template<class T, class U>
void heapsort(u32 n, T* v, U func)
{
    BVH_ASSERT(0 <= n);
    --v;
    u32 i, j;
    T x;
    for(u32 k = n >> 1; k >= 1; --k) {
        i = k;
        x = v[k];
        while((j = i << 1) <= n) {
            if(j < n && func(v[j], v[j + 1])) {
                ++j;
            }

            if(!func(x, v[j])) {
                break;
            }
            v[i] = v[j];
            i = j;
        }
        v[i] = x;
    }

    while(n > 1) {
        x = v[n];
        v[n] = v[1];
        --n;
        i = 1;
        while((j = i << 1) <= n) {
            if(j < n && func(v[j], v[j + 1])) {
                ++j;
            }

            if(!func(x, v[j])) {
                break;
            }
            v[i] = v[j];
            i = j;
        }
        v[i] = x;
    }
}

template<class T>
void heapsort(u32 n, T* v)
{
    heapsort(n, v, less<T>);
}

//--- quicksort
//------------------------------------------------
/**
U: bool operator(const T& a, const T& b) const{ return a<b;}
*/
template<class T, class U>
void quicksort(u32 n, T* v, U func)
{
    static const u32 SwitchN = 47;
    if(n < SwitchN) {
        insertionsort(n, v, func);
        return;
    }

    u32 i0 = 0;
    u32 i1 = n - 1;

    T pivot = v[(i0 + i1) >> 1];

    for(;;) {
        while(func(v[i0], pivot)) {
            ++i0;
        }

        while(func(pivot, v[i1])) {
            --i1;
        }

        if(i1 <= i0) {
            break;
        }
        bvh::swap(v[i0], v[i1]);
        ++i0;
        --i1;
    }

    if(1 < i0) {
        quicksort(i0, v, func);
    }

    ++i1;
    n = n - i1;
    if(1 < n) {
        quicksort(n, v + i1, func);
    }
}

template<class T>
void quicksort(u32 n, T* v)
{
    quicksort(n, v, less<T>);
}

//--- introsort
//------------------------------------------------
/**
 */
template<class T, class U>
void introsort(u32 n, T* v, u32 depth, U func)
{
    static const u32 SwitchN = 47;
    if(n < SwitchN) {
        insertionsort(n, v, func);
        return;
    }
    if(depth <= 0) {
        heapsort(n, v, func);
        return;
    }

    u32 i0 = 0;
    u32 i1 = n - 1;

    T pivot = v[(i0 + i1) >> 1];

    for(;;) {
        while(func(v[i0], pivot)) {
            ++i0;
        }

        while(func(pivot, v[i1])) {
            --i1;
        }

        if(i1 <= i0) {
            break;
        }
        bvh::swap(v[i0], v[i1]);
        ++i0;
        --i1;
    }

    --depth;
    if(1 < i0) {
        introsort(i0, v, depth, func);
    }

    ++i1;
    n = n - i1;
    if(1 < n) {
        introsort(n, v + i1, depth, func);
    }
}

template<class T, class U>
void introsort(u32 n, T* v, U func)
{
    u32 depth = 0;
    u32 t = n;
    while(1 < t) {
        ++depth;
        t >>= 1;
    }
    introsort(n, v, depth, func);
}

template<class T>
void introsort(u32 n, T* v)
{
    introsort(n, v, less<T>);
}

//--- specific sort
//--------------------------------------------------------------
struct SortFuncCentroid
{
    SortFuncCentroid(const f32* centroids)
        : centroids_(centroids)
    {
    }

    bool operator()(u32 i0, u32 i1) const
    {
        return centroids_[i0] < centroids_[i1];
    }

    const f32* centroids_;
};

inline void sort_centroids(u32 numPrimitives, u32* primitiveIndices, const f32* centroids)
{
    bvh::introsort(numPrimitives, primitiveIndices, SortFuncCentroid(centroids));
}

inline void insertionsort_centroids(u32 numPrimitives, u32* primitiveIndices, const f32* centroids)
{
    bvh::insertionsort(numPrimitives, primitiveIndices, SortFuncCentroid(centroids));
}

//--- Array
//--------------------------------------------------------------
template<class T, u32 Increment = 128U>
class Array
{
public:
    using this_type = Array<T, Increment>;
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = u32;
    static constexpr u32 Align = 128;
    static constexpr u32 Mask = Align - 1;

    /**
      @return if lhs<rhs then true else false
      */
    typedef bool (*SortCmp)(const T& lhs, const T& rhs);

    Array();
    Array(this_type&& rhs);
    explicit Array(size_type capacity);
    ~Array();

    inline size_type capacity() const;
    inline size_type size() const;

    inline T& operator[](u32 index);
    inline const T& operator[](u32 index) const;

    void push_back(const value_type& t);
    void pop_back();

    inline iterator begin();
    inline const_iterator begin() const;

    inline iterator end();
    inline const_iterator end() const;

    void clear();
    void reserve(size_type capacity);
    void resize(size_type size);
    void removeAt(u32 index);
    void swap(this_type& rhs);

    this_type& operator=(this_type&& rhs);

    s32 find(const T& x) const;
    void insertionsort(const T& t, SortCmp cmp);

private:
    Array(const this_type&) = delete;
    this_type& operator=(const this_type&) = delete;

    void expand_(u32 size);

    size_type capacity_;
    size_type size_;
    value_type* items_;
};

template<class T, u32 Increment>
Array<T, Increment>::Array()
    : capacity_(0)
    , size_(0)
    , items_(BVH_NULL)
{
}

template<class T, u32 Increment>
Array<T, Increment>::Array(this_type&& rhs)
    : capacity_(rhs.capacity_)
    , size_(rhs.size_)
    , items_(rhs.items_)
{
    rhs.capacity_ = 0;
    rhs.size_ = 0;
    rhs.items_ = BVH_NULL;
}

template<class T, u32 Increment>
Array<T, Increment>::Array(size_type capacity)
    : capacity_((capacity + Mask) & ~Mask)
    , size_(0)
    , items_(BVH_NULL)
{
    items_ = reinterpret_cast<value_type*>(BVH_ALIGNED_MALLOC(sizeof(value_type) * capacity_));
}

template<class T, u32 Increment>
Array<T, Increment>::~Array()
{
    BVH_ALIGNED_FREE(items_);
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::size_type
Array<T, Increment>::capacity() const
{
    return capacity_;
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::size_type
Array<T, Increment>::size() const
{
    return size_;
}

template<class T, u32 Increment>
inline T& Array<T, Increment>::operator[](u32 index)
{
    BVH_ASSERT(index < size_);
    return items_[index];
}

template<class T, u32 Increment>
inline const T& Array<T, Increment>::operator[](u32 index) const
{
    BVH_ASSERT(index < size_);
    return items_[index];
}

template<class T, u32 Increment>
void Array<T, Increment>::push_back(const value_type& t)
{
    if(capacity_ <= size_) {
        expand_();
    }
    items_[size_] = t;
    ++size_;
}

template<class T, u32 Increment>
void Array<T, Increment>::pop_back()
{
    BVH_ASSERT(0 < size_);
    --size_;
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::iterator Array<T, Increment>::begin()
{
    return items_;
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::const_iterator Array<T, Increment>::begin() const
{
    return items_;
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::iterator Array<T, Increment>::end()
{
    return items_ + size_;
}

template<class T, u32 Increment>
inline typename Array<T, Increment>::const_iterator Array<T, Increment>::end() const
{
    return items_ + size_;
}

template<class T, u32 Increment>
void Array<T, Increment>::clear()
{
    size_ = 0;
}

template<class T, u32 Increment>
void Array<T, Increment>::reserve(size_type capacity)
{
    if(capacity <= capacity_) {
        return;
    }
    expand_(capacity);
}

template<class T, u32 Increment>
void Array<T, Increment>::resize(size_type size)
{
    reserve(size);
    size_ = size;
}

template<class T, u32 Increment>
void Array<T, Increment>::removeAt(u32 index)
{
    BVH_ASSERT(index < size_);
    for(u32 i = index + 1; i < size_; ++i) {
        items_[i - 1] = items_[i];
    }
    --size_;
}

template<class T, u32 Increment>
void Array<T, Increment>::swap(this_type& rhs)
{
    bvh::swap(capacity_, rhs.capacity_);
    bvh::swap(size_, rhs.size_);
    bvh::swap(items_, rhs.items_);
}

template<class T, u32 Increment>
typename Array<T, Increment>::this_type& Array<T, Increment>::operator=(this_type&& rhs)
{
    if(this == &rhs) {
        return *this;
    }
    BVH_ALIGNED_FREE(items_);

    capacity_ = rhs.capacity_;
    size_ = rhs.size_;
    items_ = rhs.items_;

    rhs.capacity_ = 0;
    rhs.size_ = 0;
    rhs.items_ = NULL;
    return *this;
}

template<class T, u32 Increment>
s32 Array<T, Increment>::find(const T& ptr) const
{
    for(s32 i = 0; i < size_; ++i) {
        if(ptr == items_[i]) {
            return i;
        }
    }
    return -1;
}

template<class T, u32 Increment>
void Array<T, Increment>::insertionsort(const T& t, SortCmp cmp)
{
    s32 size = size_;
    push_back(t);
    for(s32 i = size - 1; 0 <= i; --i) {
        if(cmp(items_[i + 1], items_[i])) {
            bvh::swap(items_[i + 1], items_[i]);
            continue;
        }
        break;
    }
}

template<class T, u32 Increment>
void Array<T, Increment>::expand_(u32 size)
{
    u32 capacity = capacity_ + Align;
    while(capacity < size) {
        capacity += Align;
    }
    value_type* items = reinterpret_cast<value_type*>(BVH_ALIGNED_MALLOC(sizeof(value_type) * capacity));
    if(0 < capacity_) {
        BVH_MEMCPY(items, items_, sizeof(value_type) * capacity_);
    }
    BVH_ALIGNED_FREE(items_);
    capacity_ = capacity;
    items_ = items;
}

//--- RGB
//--------------------------------------------------------------
struct RGB
{
    f32 r_;
    f32 g_;
    f32 b_;
    f32 x_;
};
#    if defined(BVH_UE)
#    else
void printImage(const char* filename, RGB* rgb, u32 width, u32 height);
#    endif

//--- Vector3
//--------------------------------------------------------------
class Vector3
{
public:
    void zero();

    f32 operator[](s32 index) const
    {
        return reinterpret_cast<const f32*>(this)[index];
    }

    f32& operator[](s32 index)
    {
        return reinterpret_cast<f32*>(this)[index];
    }

    f32 length() const;
    f32 halfArea() const;

    Vector3& operator*=(f32 a);

    f32 x_, y_, z_;
};

Vector3 operator+(const Vector3& v0, const Vector3& v1);
Vector3 operator-(const Vector3& v0, const Vector3& v1);
Vector3 operator*(const Vector3& v, f32 a);
Vector3 operator*(f32 a, const Vector3& v);

f32 distance(const Vector3& p0, const Vector3& p1);
Vector3 normalize(const Vector3& v);
f32 dot(const Vector3& v0, const Vector3& v1);
Vector3 cross(const Vector3& v0, const Vector3& v1);

//--- Vector4
//--------------------------------------------------------------
class Vector4
{
public:
    f32 x_, y_, z_, w_;
};

//--- Morton Code
//--------------------------------------------------------------
/**
  @brief Generate morton code 10 bits for each axis
  */
u32 mortonCode3(u32 x, u32 y, u32 z);
/**
  @brief Reconstruct each axis' values from morton code
  */
void rmortonCode3(u32& x, u32& y, u32& z, u32 w);

struct Face
{
    Vector3 p0_;
    Vector3 p1_;
    Vector3 p2_;
};

//--- AABB
//--------------------------------------------------------------
struct AABB
{
    void zero();
    void setInvalid();

    Vector3 extent() const;
    Vector3 diagonal() const;

    void extend(const AABB& bbox);
    s32 maxExtentAxis() const;

    f32 halfArea() const;

    Vector3 bmin_;
    Vector3 bmax_;
};

//--- Sphere
//--------------------------------------------------------------
struct Sphere
{
    Vector3 center_;
    f32 radius_;
};

//--- Capsule
//--------------------------------------------------------------
struct Capsule
{
    Vector3 p0_;
    Vector3 p1_;
    f32 radius_;
};

//--- Ray
//--------------------------------------------------------------
struct Ray
{
    static Ray construct(const Vector3& origin, const Vector3& direction, f32 t);
    void invertDirection();

    void setDirection(const Vector3& direction);
    void setDirection(const Vector3& direction, const Vector3& invDirection);

    Vector3 origin_;
    Vector3 direction_;
    Vector3 invDirection_;
    f32 t_;
};

//--- HitRecord
//--------------------------------------------------------------
struct HitRecord
{
    static constexpr u32 Invalid = static_cast<u32>(-1);
    f32 t_;
    u32 primitive_;
};

struct HitRecordSet
{
    static constexpr u32 MaxHits = 4;
    struct Record
    {
        Vector3 direction_;
        f32 depth_;
        u32 primitive_;
    };
    u32 count_;
    Record records_[MaxHits];
};

bool testRay(f32& t, const Ray& ray, const Vector3& p0, const Vector3& p1, const Vector3& p2);
bool testRay(f32& tmin, f32& tmax, const Ray& ray, const AABB& aabb);

f32 sqrDistancePointSegment(const Vector3& start, const Vector3& end, const Vector3& point);
void closestPointSegment(const Vector3& closestPoint, const Vector3& start, const Vector3& end, const Vector3& point);
f32 testSphereCapsule(const Sphere& sphere, const Capsule& capsule);

bool testSphereTriangle(HitRecordSet::Record& record, const Sphere& sphere, const Vector3& p0, const Vector3& p1, const Vector3& p2);

namespace qbvh
{
    //-----------------------------------------------------------
    // Collide segment vs aabb
    s32 testRayAABB(
        vector4_t tmin,
        vector4_t tmax,
        vector4_t origin[3],
        vector4_t invDir[3],
        const s32 sign[3],
        const Vector4 bbox[2][3]);

    // Collide aabb vs aabb
    s32 testAABB(const vector4_t bbox0[2][3], const vector4_t bbox1[2][3]);

    // Collide sphre vs aabb
    s32 testSphereAABB(const vector4_t position[3], const vector4_t radius, const Vector4 bbox[2][3]);
} // namespace qbvh

//--- BinQBVH
//--------------------------------------------------------------
class BinQBVH
{
public:
    static constexpr f32 Epsilon = 1.0e-6f;
    static constexpr u32 MinLeafPrimitives = 15;
    static constexpr u32 NumBins = 32;
    static constexpr u32 MaxBinningDepth = 11;
    static constexpr u32 MaxDepth = 24;
    static constexpr u32 MaxNodes = 0xFFFFFF - 4;
    static constexpr f32 Log4 = 0.60205999132f;

    struct Joint
    {
        Vector4 bbox_[2][3];
        u32 children_;
        u8 axis0_;
        u8 axis1_;
        u8 axis2_;
        u8 flags_;
    };

    struct Leaf
    {
        u32 padding0_[22];
        u32 start_;
        u32 size_;
        u32 children_;
        u8 axis0_;
        u8 axis1_;
        u8 axis2_;
        u8 flags_;
    };

    union Node
    {
        static const u8 LeafFlag = (0x01U << 7);

        bool isLeaf() const
        {
            return LeafFlag == (leaf_.flags_ & LeafFlag);
        }

        void setLeaf(u32 start, u32 size)
        {
            leaf_.flags_ = LeafFlag;
            leaf_.start_ = start;
            leaf_.size_ = size;
        }

        void setJoint(s32 child, const AABB bbox[4], u8 axis[3]);

        u32 getPrimitiveIndex() const
        {
            return leaf_.start_;
        }

        u32 getNumPrimitives() const
        {
            return leaf_.size_;
        }

        Joint joint_;
        Leaf leaf_;
    };

    struct Work
    {
        u32 start_;
        u32 numPrimitives_;
        u32 node_;
        u32 depth_;
        AABB bbox_;
    };

    BinQBVH();
    ~BinQBVH();

    void resize(u32 size);
    Face& face(u32 index);

    void build();
    HitRecord intersect(Ray& ray);
    HitRecordSet intersect(const Sphere& sphere);
    u32 getDepth() const
    {
        return depth_;
    }

#    if defined(BVH_UE)
#    else
#        ifdef _DEBUG
    void print(const char* filename);
#        endif
#    endif

private:
    BinQBVH(const BinQBVH&) = delete;
    BinQBVH& operator=(const BinQBVH&) = delete;

    static constexpr u32 MaxWorks = MaxDepth << 2;

    static Vector3 getCentroid(const Vector3& x0, const Vector3& x1, const Vector3& x2);
    static void getBBox(AABB& bbox, const Vector3& x0, const Vector3& x1, const Vector3& x2);

    void getBBox(AABB& bbox, u32 start, u32 end);

    void recursiveConstruct(u32 numTriangles, const AABB& bbox);
    void split(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, f32 invArea, u32 start, u32 numPrimitives, const AABB& bbox);
    void splitMid(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, u32 start, u32 numPrimitives, const AABB& bbox);
    void splitBinned(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, f32 area, u32 start, u32 numPrimitives, const AABB& bbox);

    f32 SAH_KI_;
    f32 SAH_KT_;
    u32 numFaces_;
    Face* faces_;

    AABB bbox_;
    u32 depth_;
    Array<Node> nodes_;
    Array<u32> primitiveIndices_;
    Array<f32> primitiveCentroids_;
    Array<AABB> primitiveBBoxes_;
    Work works_[MaxWorks];
};
} // namespace bvh
#endif // INC_BVH_BINQBVH_H__
