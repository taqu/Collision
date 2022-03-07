/**
@file BinQBVH.cpp
@author t-sakai
@date 2018/01/22 create
*/
#include "BinQBVH.h"
#if defined(BVH_UE)
#else
#    include <cmath>
#    include <cstdio>
#endif

#if defined(BVH_NEON)
bvh_int32_t movemask(uint32x4_t x)
{
    bvh_uint32_t mask[4];
    vst1q_u32(mask, x);
    return ((mask[3] & 0x01U) << 3) | ((mask[2] & 0x01U) << 2) | ((mask[1] & 0x01U) << 1) | ((mask[0] & 0x01U) << 0);
}
#endif

namespace bvh
{
f32 clamp01(f32 v)
{
    s32* t = (s32*)&v;
    s32 s = (*t) >> 31;
    s = ~s;
    *t &= s;

    v -= 1.0f;
    s = (*t) >> 31;
    *t &= s;
    v += 1.0f;
    return v;
}

u32 leadingzero(u32 x)
{
#if defined(_MSC_VER)
    unsigned long n;
    _BitScanReverse(&n, x);
    return 31 - n;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    u32 n = 0;
    if(x <= 0x0000FFFFU) {
        n += 16;
        x <<= 16;
    }
    if(x <= 0x00FFFFFFU) {
        n += 8;
        x <<= 8;
    }
    if(x <= 0x0FFFFFFFU) {
        n += 4;
        x <<= 4;
    }
    if(x <= 0x3FFFFFFFU) {
        n += 2;
        x <<= 2;
    }
    if(x <= 0x7FFFFFFFU) {
        ++n;
    }
    return n;
#endif
}

#if defined(BVH_UE)
#else
//--- RGB
//--------------------------------------------------------------
void printImage(const char* filename, RGB* rgb, u32 width, u32 height)
{
    BVH_ASSERT(BVH_NULL != filename);
    BVH_ASSERT(BVH_NULL != rgb);

    FILE* file = BVH_NULL;
#    if defined(_MSC_VER)
    fopen_s(&file, filename, "wb");
#    else
    file = fopen(filename, "wb");
#    endif
    if(NULL == file) {
        return;
    }

    fprintf(file, "P3\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    for(u32 i = 0; i < height; ++i) {
        for(u32 j = 0; j < width; ++j) {
            const RGB& pixel = rgb[width * i + j];

            u8 r = static_cast<u8>(255 * clamp01(pixel.r_));
            u8 g = static_cast<u8>(255 * clamp01(pixel.g_));
            u8 b = static_cast<u8>(255 * clamp01(pixel.b_));
            fprintf(file, "%d %d %d ", r, g, b);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
#endif

//--- Vector3
//--------------------------------------------------------------
void Vector3::zero()
{
    x_ = y_ = z_ = 0.0f;
}

Vector3 Vector3::operator-() const
{
    return {-x_, -y_, -z_};
}

f32 Vector3::length() const
{
    return BVH_SQRT(x_ * x_ + y_ * y_ + z_ * z_);
}

f32 Vector3::halfArea() const
{
    return x_ * y_ + y_ * z_ + z_ * x_;
}

Vector3& Vector3::operator*=(f32 x)
{
    x_ *= x;
    y_ *= x;
    z_ *= x;
    return *this;
}

Vector3& Vector3::operator/=(f32 x)
{
    f32 inv = 1.0f / x;
    x_ *= inv;
    y_ *= inv;
    z_ *= inv;
    return *this;
}

Vector3 operator+(const Vector3& v0, const Vector3& v1)
{
    return {v0.x_ + v1.x_, v0.y_ + v1.y_, v0.z_ + v1.z_};
}

Vector3 operator-(const Vector3& v0, const Vector3& v1)
{
    return {v0.x_ - v1.x_, v0.y_ - v1.y_, v0.z_ - v1.z_};
}

Vector3 operator*(const Vector3& v, f32 a)
{
    return {a * v.x_, a * v.y_, a * v.z_};
}

Vector3 operator*(f32 a, const Vector3& v)
{
    return {a * v.x_, a * v.y_, a * v.z_};
}

f32 distance(const Vector3& p0, const Vector3& p1)
{
    Vector3 d = p1 - p0;
    return BVH_SQRT(dot(d, d));
}

Vector3 normalize(const Vector3& v)
{
    f32 il = 1.0f / v.length();
    return {v.x_ * il, v.y_ * il, v.z_ * il};
}

Vector3 normalizeZero(const Vector3& v)
{
    f32 l = v.length();
    if(l <= F32_EPSILON) {
        return {0.0f, 0.0f, 0.0f};
    }
    f32 il = 1.0f / l;
    return {v.x_ * il, v.y_ * il, v.z_ * il};
}

f32 dot(const Vector3& v0, const Vector3& v1)
{
    return v0.x_ * v1.x_ + v0.y_ * v1.y_ + v0.z_ * v1.z_;
}

Vector3 cross(const Vector3& v0, const Vector3& v1)
{
    f32 x = v0.y_ * v1.z_ - v0.z_ * v1.y_;
    f32 y = v0.z_ * v1.x_ - v0.x_ * v1.z_;
    f32 z = v0.x_ * v1.y_ - v0.y_ * v1.x_;
    return {x, y, z};
}

void orthonormalBasis(Vector3& xaxis, Vector3& yaxis, const Vector3& zaxis)
{
    Vector3 up;
    if(0.9999999f < absolute(zaxis.y_)) {
        up = {zaxis.x_, zaxis.z_, -zaxis.y_};
    } else {
        up = {0.0f, 1.0f, 0.0f};
    }
    xaxis = normalizeZero(cross(up, zaxis));
    yaxis = normalizeZero(cross(zaxis, xaxis));
}

namespace
{
    u32 separateBy2(u32 x)
    {
        x = (x | (x << 8) | (x << 16)) & 0x0300F00FU;
        x = (x | (x << 4) | (x << 8)) & 0x030C30C3U;
        x = (x | (x << 2) | (x << 4)) & 0x09249249U;
        return x;
    }

    u32 combineBy2(u32 x)
    {
        x &= 0x09249249U;
        x = (x | (x >> 2) | (x >> 4)) & 0x030C30C3U;
        x = (x | (x >> 4) | (x >> 8)) & 0x0300F00FU;
        x = (x | (x >> 8) | (x >> 16)) & 0x3FFU;
        return x;
    }
} // namespace
u32 mortonCode3(u32 x, u32 y, u32 z)
{
    return separateBy2(x) | (separateBy2(y) << 1) | (separateBy2(z) << 2);
}

void rmortonCode3(u32& x, u32& y, u32& z, u32 w)
{
    x = combineBy2(w);
    y = combineBy2(w >> 1);
    z = combineBy2(w >> 2);
}

void AABB::zero()
{
    bmin_.zero();
    bmax_.zero();
}

void AABB::setInvalid()
{
    bmin_.x_ = bmin_.y_ = bmin_.z_ = bvh_limits_max;
    bmax_.x_ = bmax_.y_ = bmax_.z_ = -bvh_limits_max;
}

Vector3 AABB::extent() const
{
    return bmax_ - bmin_;
}

Vector3 AABB::diagonal() const
{
    return {
        bmax_.x_ - bmin_.x_,
        bmax_.y_ - bmin_.y_,
        bmax_.z_ - bmin_.z_};
}

void AABB::extend(const AABB& bbox)
{
    bmin_.x_ = minimum(bmin_.x_, bbox.bmin_.x_);
    bmin_.y_ = minimum(bmin_.y_, bbox.bmin_.y_);
    bmin_.z_ = minimum(bmin_.z_, bbox.bmin_.z_);

    bmax_.x_ = maximum(bmax_.x_, bbox.bmax_.x_);
    bmax_.y_ = maximum(bmax_.y_, bbox.bmax_.y_);
    bmax_.z_ = maximum(bmax_.z_, bbox.bmax_.z_);
}

s32 AABB::maxExtentAxis() const
{
    Vector3 extent = bmax_ - bmin_;
    s32 axis = (extent.x_ < extent.y_) ? 1 : 0;
    axis = (extent.z_ < extent[axis]) ? axis : 2;
    return axis;
}

f32 AABB::halfArea() const
{
    f32 dx = bmax_.x_ - bmin_.x_;
    f32 dy = bmax_.y_ - bmin_.y_;
    f32 dz = bmax_.z_ - bmin_.z_;
    return (dx * dy + dy * dz + dz * dx);
}

//--- Plane
//--------------------------------------------------------------
void Plane::set(const Vector3& point, const Vector3& normal)
{
    nx_ = normal.x_;
    ny_ = normal.y_;
    nz_ = normal.z_;
    d_ = -dot(point, normal);
}

f32 Plane::distance(const Vector3& point) const
{
    return nx_ * point.x_ + ny_ * point.y_ + nz_ * point.z_ + d_;
}

//--- Ray
//--------------------------------------------------------------
Ray Ray::construct(const Vector3& origin, const Vector3& direction, f32 t)
{
    Ray ray;
    ray.origin_ = origin;
    ray.direction_ = direction;
    ray.t_ = t;
    ray.invertDirection();
    return ray;
}

void Ray::invertDirection()
{
    for(s32 i = 0; i < 3; ++i) {
        if(0.0f <= direction_[i]) {
            invDirection_[i] = (isZeroPositive(direction_[i])) ? bvh_limits_max : 1.0f / direction_[i];
        } else {
            invDirection_[i] = (isZeroNegative(direction_[i])) ? -bvh_limits_max : 1.0f / direction_[i];
        }
    }
}

void Ray::setDirection(const Vector3& direction)
{
    direction_ = direction;
    invertDirection();
}

void Ray::setDirection(const Vector3& direction, const Vector3& invDirection)
{
    direction_ = direction;
    invDirection_ = invDirection;
}

bool testRay(f32& t, const Ray& ray, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
    Vector3 d0 = p1 - p0;
    Vector3 d1 = p2 - p0;
    Vector3 c = cross(ray.direction_, d1);

    Vector3 tvec;
    f32 discr = dot(c, d0);
    Vector3 qvec;
    if(F32_EPSILON < discr) {
        // Front face
        tvec = ray.origin_ - p0;
        f32 v = dot(tvec, c);
        if(v < 0.0f || discr < v) {
            return false;
        }
        qvec = cross(tvec, d0);
        f32 w = dot(qvec, ray.direction_);
        if(w < 0.0f || discr < (v + w)) {
            return false;
        }

    } else if(discr < -F32_EPSILON) {
        // Back face
        tvec = ray.origin_ - p0;
        f32 v = dot(tvec, c);
        if(0.0f < v || v < discr) {
            return false;
        }
        qvec = cross(tvec, d0);
        f32 w = dot(qvec, ray.direction_);
        if(0.0f < w || (v + w) < discr) {
            return false;
        }

    } else {
        // Parallel with the plane
        return false;
    }

    f32 invDiscr = 1.0f / discr;

    t = dot(d1, qvec);
    t *= invDiscr;
    return true;
}

bool testRay(f32& tmin, f32& tmax, const Ray& ray, const AABB& aabb)
{
    tmin = 0.0f;
    tmax = ray.t_;

    for(s32 i = 0; i < 3; ++i) {
        if(absolute(ray.direction_[i]) < F32_HITEPSILON) {
            if(ray.origin_[i] < aabb.bmin_[i] || aabb.bmax_[i] < ray.origin_[i]) {
                return false;
            }

        } else {
            f32 invD = ray.invDirection_[i];
            f32 t1 = (aabb.bmin_[i] - ray.origin_[i]) * invD;
            f32 t2 = (aabb.bmax_[i] - ray.origin_[i]) * invD;

            if(t1 > t2) {
                if(t2 > tmin)
                    tmin = t2;
                if(t1 < tmax)
                    tmax = t1;
            } else {
                if(t1 > tmin)
                    tmin = t1;
                if(t2 < tmax)
                    tmax = t2;
            }

            if(tmin > tmax) {
                return false;
            }
            if(tmax < 0.0f) {
                return false;
            }
        }
    }
    return true;
}

bool testRayAABB(f32& tmin, const Vector3& origin, const Vector3& direction, const AABB& aabb)
{
    tmin = 0.0f;
    f32 tmax = bvh_limits_max;
    for(u32 i = 0; i < 3; ++i) {
        if(absolute(direction[i]) < F32_EPSILON) {
            if(origin[i] < aabb.bmin_[i] || aabb.bmax_[i] < origin[i]) {
                return false;
            }
        } else {
            f32 invD = 1.0f / direction[i];
            f32 t1 = (aabb.bmin_[i] - origin[i]) * invD;
            f32 t2 = (aabb.bmax_[i] - origin[i]) * invD;
            if(t1 > t2) {
                if(t2 > tmin)
                    tmin = t2;
                if(t1 < tmax)
                    tmax = t1;
            } else {
                if(t1 > tmin)
                    tmin = t1;
                if(t2 < tmax)
                    tmax = t2;
            }

            if(tmin > tmax) {
                return false;
            }
            if(tmax < 0.0f) {
                return false;
            }
        }
    }
    return true;
}

f32 sqrDistancePointSegment(const Vector3& start, const Vector3& end, const Vector3& point)
{
    Vector3 d0 = end - start;
    Vector3 d1 = point - start;
    Vector3 d2 = point - end;
    f32 e = dot(d0, d1);
    if(e < -F32_EPSILON) {
        return dot(d1, d1);
    }
    f32 f = dot(d0, d0);
    if(f <= e) {
        return dot(d2, d2);
    }
    return dot(d1, d1) - e * e / f;
}

void closestPointSegment(Vector3& closestPoint, const Vector3& p0, const Vector3& p1, const Vector3& point)
{
    Vector3 direction = p1 - p0;
    f32 t = dot(point - p0, direction);
    if(t < -F32_EPSILON) {
        closestPoint = p0;
    } else {
        f32 denom = dot(direction, direction);
        if(denom <= t) {
            closestPoint = p1;
        } else {
            t = t / denom;
            closestPoint = p0 + (direction * t);
        }
    }
}

f32 closestSegmentSegment(Vector3& c0, Vector3& c1, const Vector3& p0, const Vector3& p1, const Vector3& q0, const Vector3& q1)
{
    Vector3 d0 = q0 - p0;
    Vector3 d1 = q1 - p1;
    Vector3 r = p0 - p1;
    f32 a = dot(d0, d0);
    f32 e = dot(d1, d1);
    f32 f = dot(d1, r);
    f32 s, t;
    if(a <= F32_EPSILON) {
        s = 0.0f;
        t = f / e;
        t = bvh::clamp01(t);
    } else {
        f32 c = dot(d0, r);
        if(e <= F32_EPSILON) {
            t = 0.0f;
            s = bvh::clamp01(-c / a);
        } else {
            f32 b = dot(d0, d1);
            f32 denom = a * e - b * b;
            if(0.0f < denom) {
                s = bvh::clamp01((b * f - c * e) / denom);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if(t < 0.0f) {
                t = 0.0f;
                s = bvh::clamp01(-c / a);
            } else if(1.0f < t) {
                t = 1.0f;
                s = bvh::clamp01((b - c) / a);
            }
        }
    }
    c0 = p0 + d0 * s;
    c1 = p1 + d1 * t;
    Vector3 d2 = c0 - c1;
    return dot(d2, d2);
}

f32 sqrDistanceSegmentSegment(const Vector3& p0, const Vector3& p1, const Vector3& q0, const Vector3& q1)
{
    Vector3 d0 = q0 - p0;
    Vector3 d1 = q1 - p1;
    Vector3 r = p0 - p1;
    f32 a = dot(d0, d0);
    f32 e = dot(d1, d1);
    f32 f = dot(d1, r);
    f32 s, t;
    if(a <= F32_EPSILON) {
        s = 0.0f;
        t = f / e;
        t = bvh::clamp01(t);
    } else {
        f32 c = dot(d0, r);
        if(e <= F32_EPSILON) {
            t = 0.0f;
            s = bvh::clamp01(-c / a);
        } else {
            f32 b = dot(d0, d1);
            f32 denom = a * e - b * b;
            if(0.0f < denom) {
                s = bvh::clamp01((b * f - c * e) / denom);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if(t < 0.0f) {
                t = 0.0f;
                s = bvh::clamp01(-c / a);
            } else if(1.0f < t) {
                t = 1.0f;
                s = bvh::clamp01((b - c) / a);
            }
        }
    }
    Vector3 c0 = p0 + d0 * s;
    Vector3 c1 = p1 + d1 * t;
    Vector3 d2 = c0 - c1;
    return dot(d2, d2);
}

f32 testSphereCapsule(const Sphere& sphere, const Capsule& capsule)
{
    f32 d2 = sqrDistancePointSegment(capsule.p0_, capsule.p1_, sphere.center_);
    return sphere.radius_ - BVH_SQRT(d2);
}

f32 testCapsuleCapsule(const Capsule& capsule0, const Capsule& capsule1)
{
    Vector3 c0, c1;
    f32 d2 = closestSegmentSegment(c0, c1, capsule0.p0_, capsule0.p1_, capsule1.p0_, capsule1.p1_);
    f32 radius = capsule0.radius_ + capsule1.radius_;
    return radius - BVH_SQRT(d2);
}

bool testSegmentCapsule(f32& t, const Vector3& p0, const Vector3& p1, const Capsule& capsule)
{
    Vector3 c0, c1;
    f32 d2 = closestSegmentSegment(c0, c1, p0, p1, capsule.p0_, capsule.p1_);
    f32 radius = capsule.radius_;
    t = radius - BVH_SQRT(d2);
    return 0.0f <= t;
}

bool testSphereTriangle(f32& t, const Sphere& sphere, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
    // Test if sphere lies outside the triangle plane
    // Translate to origin
    Vector3 A = p0 - sphere.center_;
    Vector3 B = p1 - sphere.center_;
    Vector3 C = p2 - sphere.center_;
    f32 rr = sphere.radius_ * sphere.radius_;

    Vector3 AB = B - A;
    Vector3 V = cross(AB, C - A);
    f32 d = dot(A, V);
    f32 e = dot(V, V);
    bool sep0 = (rr * e) < (d * d);

    // Test if sphere lies outside a triangle vertex
    f32 aa = dot(A, A);
    f32 bb = dot(B, B);
    f32 cc = dot(C, C);
    f32 ab = dot(A, B);
    f32 ac = dot(A, C);
    f32 bc = dot(B, C);
    bool sep1 = (rr < aa) && (aa < ab) && (aa < ac);
    bool sep2 = (rr < bb) && (bb < ab) && (bb < bc);
    bool sep3 = (rr < cc) && (cc < ac) && (cc < bc);

    // Test if sphre lies outside a triangle edge
    Vector3 BC = C - B;
    Vector3 CA = A - C;
    f32 d0 = ab - aa;
    f32 d1 = bc - bb;
    f32 d2 = ac - cc;
    f32 e0 = dot(AB, AB);
    f32 e1 = dot(BC, BC);
    f32 e2 = dot(CA, CA);
    Vector3 Q0 = A * e0 - d0 * AB;
    Vector3 Q1 = B * e1 - d1 * BC;
    Vector3 Q2 = C * e2 - d2 * CA;
    Vector3 QC = C * e0 - Q0;
    Vector3 QA = A * e1 - Q1;
    Vector3 QB = B * e2 - Q2;
    bool sep4 = ((e0 * e0 * rr) < dot(Q0, Q0)) && (0.0f < dot(Q0, QC));
    bool sep5 = ((e1 * e1 * rr) < dot(Q1, Q1)) && (0.0f < dot(Q1, QA));
    bool sep6 = ((e2 * e2 * rr) < dot(Q2, Q2)) && (0.0f < dot(Q2, QB));

    if(sep0 || sep1 || sep2 || sep3 || sep4 || sep5 || sep6) {
        Vector3 normal = normalize(V);
        t = sphere.radius_ - dot(sphere.center_ - p0, normal);
        return true;
    } else {
        return false;
    }
}

bool testCapsuleTriangle(f32& t, Vector3& point, const Capsule& capsule, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
    Vector3 e0 = p1 - p0;
    Vector3 e1 = p2 - p1;
    Vector3 e2 = p0 - p2;
    Vector3 normal = normalize(cross(e0, -e2));
    Vector3 direction = normalize(capsule.p1_ - capsule.p0_);

    Vector3 reference;
    f32 cosine = dot(normal, direction);
    if(-F32_EPSILON <= cosine && cosine <= F32_EPSILON) {
        reference = p0;
    } else {
        f32 tmp = dot(normal, (p0 - capsule.p0_)) / cosine;
        Vector3 pointOnPlane = capsule.p0_ + direction * tmp;

        Vector3 d0 = pointOnPlane - p0;
        Vector3 d1 = pointOnPlane - p1;
        Vector3 d2 = pointOnPlane - p2;
        Vector3 c0 = cross(d0, e0);
        Vector3 c1 = cross(d1, e1);
        Vector3 c2 = cross(d2, e2);
        f32 t0 = dot(c0, normal);
        f32 t1 = dot(c1, normal);
        f32 t2 = dot(c2, normal);

        if(t0 <= 0.0f && t1 <= 0.0f && t2 <= 0.0f) {
            reference = pointOnPlane;
        } else {
            f32 minDistance;
            f32 distance;
            tmp = clamp01(dot(d0, e0) / dot(e0, e0));
            Vector3 point0 = p0 + tmp * e0;
            d0 = pointOnPlane - point0;
            minDistance = dot(d0, d0);
            reference = point0;

            tmp = clamp01(dot(d1, e1) / dot(e1, e1));
            Vector3 point1 = p1 + tmp * e1;
            d1 = pointOnPlane - point1;
            distance = dot(d1, d1);
            if(distance < minDistance) {
                minDistance = distance;
                reference = point1;
            }

            tmp = clamp01(dot(d2, e2) / dot(e2, e2));
            Vector3 point2 = p2 + tmp * e2;
            d2 = pointOnPlane - point2;
            distance = dot(d2, d2);
            if(distance < minDistance) {
                minDistance = distance;
                reference = point2;
            }
        }
    }

    closestPointSegment(point, capsule.p0_, capsule.p1_, reference);
    f32 r = distance(point, reference);
    if(r <= capsule.radius_) {
        if(r <= F32_EPSILON) {
            f32 h0 = dot(normal, capsule.p0_);
            f32 h1 = dot(normal, capsule.p1_);
            t = capsule.radius_ + absolute(minimum(h0, h1));
        } else {
            t = capsule.radius_ - r;
        }
        return true;
    } else {
        return false;
    }
}

bool testOBBAABB(const OBB& obb0, const AABB& aabb)
{
    OBB obb1;
    obb1.axis_[0] = {1.0f, 0.0f, 0.0f};
    obb1.axis_[1] = {0.0f, 1.0f, 0.0f};
    obb1.axis_[2] = {0.0f, 0.0f, 1.0f};
    obb1.center_ = (aabb.bmin_ + aabb.bmax_) * 0.5f;
    obb1.half_ = (aabb.bmax_ - aabb.bmin_) * 0.5f;

    f32 r0, r1;
    Vector3 R[3];
    Vector3 AbsR[3];
    for(u32 i = 0; i < 3; ++i) {
        for(u32 j = 0; j < 3; ++j) {
            R[i][j] = dot(obb0.axis_[i], obb1.axis_[j]);
            AbsR[i][j] = absolute(R[i][j]) + F32_EPSILON;
        }
    }
    Vector3 translate = obb1.center_ - obb0.center_;
    translate = {dot(translate, obb0.axis_[0]), dot(translate, obb0.axis_[1]), dot(translate, obb0.axis_[2])};

    //
    for(u32 i = 0; i < 3; ++i) {
        r0 = obb0.half_[i];
        r1 = obb1.half_[0] * AbsR[i][0] + obb1.half_[1] * AbsR[i][1] + obb1.half_[2] * AbsR[i][2];
        if((r0 + r1) < absolute(translate[i])) {
            return false;
        }
    }

    //
    for(u32 i = 0; i < 3; ++i) {
        r0 = obb0.half_[0] * AbsR[0][i] + obb0.half_[1] * AbsR[1][i] + obb0.half_[2] * AbsR[2][i];
        r1 = obb1.half_[i];
        if((r0 + r1) < absolute(translate[0] * R[0][i] + translate[1] * R[1][i] + translate[2] * R[2][i])) {
            return false;
        }
    }

    //
    r0 = obb0.half_[1] * AbsR[2][0] + obb0.half_[2] * AbsR[1][0];
    r1 = obb1.half_[1] * AbsR[0][2] + obb1.half_[2] * AbsR[0][1];
    if((r0 + r1) < absolute(translate[2] * R[1][0] - translate[1] * R[2][0])) {
        return false;
    }

    //
    r0 = obb0.half_[1] * AbsR[2][1] + obb0.half_[2] * AbsR[1][1];
    r1 = obb1.half_[0] * AbsR[0][2] + obb1.half_[2] * AbsR[0][0];
    if((r0 + r1) < absolute(translate[2] * R[1][1] - translate[1] * R[2][1])) {
        return false;
    }

    //
    r0 = obb0.half_[1] * AbsR[2][2] + obb0.half_[2] * AbsR[1][2];
    r1 = obb1.half_[0] * AbsR[0][1] + obb1.half_[1] * AbsR[0][0];
    if((r0 + r1) < absolute(translate[2] * R[1][2] - translate[1] * R[2][2])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[2][0] + obb0.half_[2] * AbsR[0][0];
    r1 = obb1.half_[1] * AbsR[1][2] + obb1.half_[2] * AbsR[1][1];
    if((r0 + r1) < absolute(translate[0] * R[2][0] - translate[2] * R[0][0])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[2][1] + obb0.half_[2] * AbsR[0][1];
    r1 = obb1.half_[0] * AbsR[1][2] + obb1.half_[2] * AbsR[1][0];
    if((r0 + r1) < absolute(translate[0] * R[2][1] - translate[2] * R[0][1])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[2][2] + obb0.half_[2] * AbsR[0][2];
    r1 = obb1.half_[0] * AbsR[1][1] + obb1.half_[1] * AbsR[1][0];
    if((r0 + r1) < absolute(translate[0] * R[2][2] - translate[2] * R[0][2])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[1][0] + obb0.half_[1] * AbsR[0][0];
    r1 = obb1.half_[1] * AbsR[2][2] + obb1.half_[2] * AbsR[2][1];
    if((r0 + r1) < absolute(translate[1] * R[0][0] - translate[0] * R[1][0])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[1][1] + obb0.half_[1] * AbsR[0][1];
    r1 = obb1.half_[0] * AbsR[2][2] + obb1.half_[2] * AbsR[2][0];
    if((r0 + r1) < absolute(translate[1] * R[0][1] - translate[0] * R[1][1])) {
        return false;
    }

    //
    r0 = obb0.half_[0] * AbsR[1][2] + obb0.half_[1] * AbsR[0][2];
    r1 = obb1.half_[0] * AbsR[2][1] + obb1.half_[1] * AbsR[2][0];
    if((r0 + r1) < absolute(translate[1] * R[0][2] - translate[0] * R[1][2])) {
        return false;
    }

    return true;
}

namespace
{
    Vector3 getCorner(const AABB& aabb, u32 n)
    {
        Vector3 point;
        point.x_ = (0 != (0x01U & n)) ? aabb.bmax_.x_ : aabb.bmin_.x_;
        point.y_ = (0 != (0x02U & n)) ? aabb.bmax_.y_ : aabb.bmin_.y_;
        point.z_ = (0 != (0x04U & n)) ? aabb.bmax_.z_ : aabb.bmin_.z_;
        return point;
    }
} // namespace

bool testCapsuleAABB(const Capsule& capsule, const AABB& aabb)
{
    AABB extended;
    extended.bmin_.x_ = aabb.bmin_.x_ - capsule.radius_;
    extended.bmin_.y_ = aabb.bmin_.y_ - capsule.radius_;
    extended.bmin_.z_ = aabb.bmin_.z_ - capsule.radius_;
    extended.bmax_.x_ = aabb.bmax_.x_ + capsule.radius_;
    extended.bmax_.y_ = aabb.bmax_.y_ + capsule.radius_;
    extended.bmax_.z_ = aabb.bmax_.z_ + capsule.radius_;
    Vector3 direction = capsule.p1_ - capsule.p0_;
    Vector3 n = normalize(direction);
    f32 tmin;
    if(!testRayAABB(tmin, capsule.p0_, n, extended) || 1.0f < tmin) {
        return false;
    }
    Vector3 point = capsule.p0_ + direction * tmin;
    u32 u = 0;
    u32 v = 0;
    if(point.x_ < aabb.bmin_.x_) {
        u |= 1U;
    }
    if(aabb.bmax_.x_ < point.x_) {
        v |= 1U;
    }
    if(point.y_ < aabb.bmin_.y_) {
        u |= 2U;
    }
    if(aabb.bmax_.y_ < point.y_) {
        v |= 2U;
    }
    if(point.z_ < aabb.bmin_.z_) {
        u |= 4U;
    }
    if(aabb.bmax_.z_ < point.z_) {
        v |= 4U;
    }
    u32 m = u + v;
    if(7U == m) {
        tmin = bvh_limits_max;
        f32 t;
        if(testSegmentCapsule(t, getCorner(aabb, v), getCorner(aabb, v ^ 1U), capsule)) {
            tmin = minimum(t, tmin);
        }
        if(testSegmentCapsule(t, getCorner(aabb, v), getCorner(aabb, v ^ 2U), capsule)) {
            tmin = minimum(t, tmin);
        }
        if(testSegmentCapsule(t, getCorner(aabb, v), getCorner(aabb, v ^ 4U), capsule)) {
            tmin = minimum(t, tmin);
        }
        if(bvh_limits_max <= tmin) {
            return false;
        }
        return true;
    }
    if(0 == (m & (m - 1))) {
        return true;
    }
    return testSegmentCapsule(tmin, getCorner(aabb, u ^ 7), getCorner(aabb, v), capsule);
}

namespace qbvh
{
    //-----------------------------------------------------------
    // Collide segment vs aabb
    u32 testRayAABB(
        vector4_t tmin,
        vector4_t tmax,
        const vector4_t origin[3],
        const vector4_t invDir[3],
        const u32 sign[3],
        const Vector4 bbox[2][3])
    {
        for(u32 i = 0; i < 3; ++i) {
            vector4_t b0 = load(reinterpret_cast<const f32*>(&bbox[sign[i]][i]));
            tmin = maximum4(
                tmin,
                mul(sub(b0, origin[i]), invDir[i]));

            vector4_t b1 = load(reinterpret_cast<const f32*>(&bbox[1U - sign[i]][i]));
            tmax = minimum4(
                tmax,
                mul(sub(b1, origin[i]), invDir[i]));
        }
        return static_cast<u32>(movemask(cmpge(tmax, tmin)));
    }

    // Collide aabb vs aabb
    u32 testAABB(const vector4_t bbox0[2][3], const Vector4 bbox1[2][3])
    {
        Mask mask;
        mask.u_ = 0xFFFFFFFFU;

        uvector4_t t = set1_u32(mask);
        for(u32 i = 0; i < 3; ++i) {
            vector4_t b0 = load(reinterpret_cast<const f32*>(&bbox0[0][i]));
            vector4_t b1 = load(reinterpret_cast<const f32*>(&bbox1[1][i]));
            t = and4(t, cmple(b0, b1));

            vector4_t b2 = load(reinterpret_cast<const f32*>(&bbox0[1][i]));
            vector4_t b3 = load(reinterpret_cast<const f32*>(&bbox1[0][i]));
            t = and4(t, cmple(b3, b2));
        }
        return static_cast<u32>(movemask(t));
    }

    // Collide sphre vs aabb
    u32 testSphereAABB(const vector4_t position[3], const vector4_t& radius, const Vector4 bbox[2][3])
    {
        Mask mask;
        mask.u_ = 0xFFFFFFFFU;

        vector4_t tbbox[2][3];
        for(u32 i = 0; i < 3; ++i) {
            vector4_t b0 = load(reinterpret_cast<const f32*>(&bbox[0][i]));
            tbbox[0][i] = sub(b0, radius);
            vector4_t b1 = load(reinterpret_cast<const f32*>(&bbox[1][i]));
            tbbox[1][i] = add(b1, radius);
        }
        uvector4_t t = set1_u32(mask);
        for(u32 i = 0; i < 3; ++i) {
            t = and4(t, cmple(position[i], tbbox[1][i]));
            t = and4(t, cmple(tbbox[0][i], position[i]));
        }
        return static_cast<u32>(movemask(t));
    }
} // namespace qbvh

//--- BinQBVH
//--------------------------------------------------------------
void BinQBVH::Node::setJoint(s32 child, const AABB bbox[4], u8 axis[3])
{
    BVH_ALIGN16 f32 bb[2][3][4];
    joint_.flags_ = 0;
    joint_.children_ = child;
    for(u32 i = 0; i < 3; ++i) {
        for(u32 j = 0; j < 4; ++j) {
            bb[0][i][j] = bbox[j].bmin_[i] - Epsilon;
            bb[1][i][j] = bbox[j].bmax_[i] + Epsilon;
        }
    }

    for(u32 i = 0; i < 2; ++i) {
        for(u32 j = 0; j < 3; ++j) {
            store((f32*)&joint_.bbox_[i][j], load(bb[i][j]));
        }
    }

    joint_.axis0_ = axis[0];
    joint_.axis1_ = axis[1];
    joint_.axis2_ = axis[2];
}

BinQBVH::BinQBVH()
    : SAH_KI_(1.5f)
    , SAH_KT_(1.0f)
    , numFaces_(0)
    , faces_(BVH_NULL)
    , depth_(0)
{
}

BinQBVH::~BinQBVH()
{
    BVH_ALIGNED_FREE(faces_);
}

void BinQBVH::resize(u32 size)
{
    BVH_ALIGNED_FREE(faces_);
    numFaces_ = size;
    faces_ = reinterpret_cast<Face*>(BVH_ALIGNED_MALLOC(sizeof(Face) * size));
}

Face& BinQBVH::face(u32 index)
{
    return faces_[index];
}

void BinQBVH::build()
{
    u32 numTriangles = numFaces_;
    f32 depth = logf(static_cast<f32>(numTriangles) / MinLeafPrimitives) / Log4;
    u32 numNodes = static_cast<u32>(powf(2.0f, depth) + 0.5f);
    nodes_.reserve(numNodes);
    nodes_.resize(1);

    primitiveIndices_.resize(numTriangles);
    primitiveCentroids_.resize(numTriangles * 3);
    primitiveBBoxes_.resize(numTriangles);

    // Reserve
    f32* centroidX = &primitiveCentroids_[0];
    f32* centroidY = centroidX + numTriangles;
    f32* centroidZ = centroidY + numTriangles;

    // Get bboxes
    bbox_.setInvalid();
    for(u32 i = 0; i < numTriangles; ++i) {
        primitiveIndices_[i] = i;

        Vector3 centroid = getCentroid(faces_[i].p0_, faces_[i].p1_, faces_[i].p2_);
        centroidX[i] = centroid.x_;
        centroidY[i] = centroid.y_;
        centroidZ[i] = centroid.z_;
        getBBox(primitiveBBoxes_[i], faces_[i].p0_, faces_[i].p1_, faces_[i].p2_);

        bbox_.extend(primitiveBBoxes_[i]);
    }

    // Build tree
    depth_ = 1;
    recursiveConstruct(numTriangles, bbox_);

    primitiveCentroids_.clear();
    primitiveBBoxes_.clear();
}

Vector3 BinQBVH::getCentroid(const Vector3& x0, const Vector3& x1, const Vector3& x2)
{
    Vector3 p = x0 + x1 + x2;
    return p * (1.0f / 3.0f);
}

void BinQBVH::getBBox(AABB& bbox, const Vector3& x0, const Vector3& x1, const Vector3& x2)
{
    bbox.bmin_ = x0;
    bbox.bmax_ = x0;
    bbox.bmin_.x_ = minimum(bbox.bmin_.x_, x1.x_);
    bbox.bmin_.y_ = minimum(bbox.bmin_.y_, x1.y_);
    bbox.bmin_.z_ = minimum(bbox.bmin_.z_, x1.z_);

    bbox.bmax_.x_ = maximum(bbox.bmax_.x_, x1.x_);
    bbox.bmax_.y_ = maximum(bbox.bmax_.y_, x1.y_);
    bbox.bmax_.z_ = maximum(bbox.bmax_.z_, x1.z_);

    bbox.bmin_.x_ = minimum(bbox.bmin_.x_, x2.x_);
    bbox.bmin_.y_ = minimum(bbox.bmin_.y_, x2.y_);
    bbox.bmin_.z_ = minimum(bbox.bmin_.z_, x2.z_);

    bbox.bmax_.x_ = maximum(bbox.bmax_.x_, x2.x_);
    bbox.bmax_.y_ = maximum(bbox.bmax_.y_, x2.y_);
    bbox.bmax_.z_ = maximum(bbox.bmax_.z_, x2.z_);
}

void BinQBVH::getBBox(AABB& bbox, u32 start, u32 end)
{
    bbox.setInvalid();
    for(u32 i = start; i < end; ++i) {
        bbox.extend(primitiveBBoxes_[primitiveIndices_[i]]);
    }
}

void BinQBVH::recursiveConstruct(u32 numTriangles, const AABB& bbox)
{
    AABB childBBox[4];
    u32 primStart[4];
    u32 num[4];
    u8 axis[4];

    s32 stack = 0;
    works_[0] = {0, numTriangles, 0, 1, bbox};
    while(0 <= stack) {
        Work work = works_[stack];
        --stack;

        depth_ = maximum(work.depth_, depth_);
        if(work.numPrimitives_ <= MinLeafPrimitives || MaxDepth <= work.depth_ || MaxNodes <= nodes_.size()) {
            nodes_[work.node_].setLeaf(work.start_, work.numPrimitives_);
            continue;
        }

        primStart[0] = work.start_;
        if(MaxBinningDepth < work.depth_) {
            // Split top
            splitMid(axis[0], num[0], num[2], childBBox[0], childBBox[2], primStart[0], work.numPrimitives_, work.bbox_);
            primStart[2] = work.start_ + num[0];

            // Split left
            splitMid(axis[1], num[0], num[1], childBBox[0], childBBox[1], work.start_, num[0], childBBox[0]);
            primStart[1] = work.start_ + num[0];

            // Split right
            splitMid(axis[2], num[2], num[3], childBBox[2], childBBox[3], primStart[2], num[2], childBBox[2]);
            primStart[3] = primStart[2] + num[2];

        } else {
            // Switch to split with the mid point, if the area is too small or the number of primitives is few.

            // Split top
            f32 area = work.bbox_.halfArea();
            if(area <= Epsilon) {
                splitMid(axis[0], num[0], num[2], childBBox[0], childBBox[2], primStart[0], work.numPrimitives_, work.bbox_);

            } else if(work.numPrimitives_ < NumBins) {
                splitMid(axis[0], num[0], num[2], childBBox[0], childBBox[2], primStart[0], work.numPrimitives_, work.bbox_);

            } else {
                splitBinned(axis[0], num[0], num[2], childBBox[0], childBBox[2], area, primStart[0], work.numPrimitives_, work.bbox_);
            }
            primStart[2] = work.start_ + num[0];

            // Split left
            area = childBBox[0].halfArea();
            if(area <= Epsilon) {
                splitMid(axis[1], num[0], num[1], childBBox[0], childBBox[1], primStart[0], num[0], childBBox[0]);

            } else if(num[0] < NumBins) {
                splitMid(axis[1], num[0], num[1], childBBox[0], childBBox[1], primStart[0], num[0], childBBox[0]);

            } else {
                splitBinned(axis[1], num[0], num[1], childBBox[0], childBBox[1], area, primStart[0], num[0], childBBox[0]);
            }
            primStart[1] = work.start_ + num[0];

            // Split right
            area = childBBox[2].halfArea();
            if(area <= Epsilon) {
                splitMid(axis[2], num[2], num[3], childBBox[2], childBBox[3], primStart[2], num[2], childBBox[2]);

            } else if(num[2] < NumBins) {
                splitMid(axis[2], num[2], num[3], childBBox[2], childBBox[3], primStart[2], num[2], childBBox[2]);

            } else {
                splitBinned(axis[2], num[2], num[3], childBBox[2], childBBox[3], area, primStart[2], num[2], childBBox[2]);
            }
            primStart[3] = primStart[2] + num[2];
        }

        if(nodes_.capacity() < (nodes_.size() + 4)) {
            nodes_.reserve(nodes_.capacity() << 1); // expand with doubling
        }

        u32 child = nodes_.size();
        nodes_[work.node_].setJoint(child, childBBox, axis);
        nodes_.resize(nodes_.size() + 4);
        for(u32 i = 0; i < 4; ++i) {
            works_[++stack] = {primStart[i], num[i], child, work.depth_ + 1, childBBox[i]};
            ++child;
        }
    }
}

// 
void BinQBVH::split(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, f32 invArea, u32 start, u32 numPrimitives, const AABB& bbox)
{
    u32 end = start + numPrimitives;
    u32 mid = start + (numPrimitives >> 1);

    f32 area_l, area_r;
    f32 bestCost = bvh_limits_max;

    // Choose the axis which has maximum extent.
    axis = static_cast<u8>(bbox.maxExtentAxis());
    f32* bestCentroids = &primitiveCentroids_[0] + axis * primitiveIndices_.size();
    bvh::insertionsort_centroids(numPrimitives, &primitiveIndices_[start], bestCentroids);

    AABB bl, br;
    for(u32 m = start + 1; m < end; ++m) {
        getBBox(bl, start, m);
        getBBox(br, m, end);

        area_l = bl.halfArea();
        area_r = br.halfArea();
        num_l = m - start;
        num_r = numPrimitives - num_l;

        f32 cost = SAH_KT_ + SAH_KI_ * invArea * (area_l * num_l + area_r * num_r);
        if(cost < bestCost) {
            mid = m;
            bestCost = cost;
            bbox_l = bl;
            bbox_r = br;
        }
    }

    num_l = mid - start;
    num_r = numPrimitives - num_l;
}

// 
void BinQBVH::splitMid(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, u32 start, u32 numPrimitives, const AABB& bbox)
{
    // split at the mid point
    axis = static_cast<u8>(bbox.maxExtentAxis());

    u32 end = start + numPrimitives;
    num_l = (numPrimitives >> 1);
    num_r = numPrimitives - num_l;
    u32 mid = start + num_l;

    f32* centroids = &primitiveCentroids_[0] + axis * primitiveIndices_.size();
    bvh::sort_centroids(numPrimitives, &primitiveIndices_[start], centroids);

    getBBox(bbox_l, start, mid);
    getBBox(bbox_r, mid, end);
}

//
void BinQBVH::splitBinned(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, f32 area, u32 start, u32 numPrimitives, const AABB& bbox)
{
    BVH_ALIGN16 u32 minBins[NumBins];
    BVH_ALIGN16 u32 maxBins[NumBins];

    vector4_t zero = setzero();

    f32 invArea = 1.0f / area;
    axis = 0;
    u32 end = start + numPrimitives;

    f32* centroids = &primitiveCentroids_[0];
    f32* bestCentroids = centroids;

    f32 bestCost = bvh_limits_max;
    u32 midBin = NumBins / 2;
    u32 step = static_cast<u32>(::log10f(static_cast<f32>(numPrimitives)));

    // Try all axis and choose the best axis and the best position
    Vector3 extent = bbox.extent();
    Vector3 unit = extent * (1.0f / NumBins); // Extents of each axis per a bin
    for(u8 curAxis = 0; curAxis < 3; ++curAxis) {
        for(u32 i = 0; i < NumBins; i += 4) {
            store(reinterpret_cast<f32*>(&minBins[i]), zero);
            store(reinterpret_cast<f32*>(&maxBins[i]), zero);
        }
        sort_centroids(numPrimitives, &primitiveIndices_[start], centroids);

        f32 invUnit = (absolute(unit[curAxis]) < Epsilon) ? 0.0f : 1.0f / unit[curAxis];
        f32 bmin = bbox.bmin_[curAxis];

        for(u32 i = start; i < end; i += step) {
            u32 index = primitiveIndices_[i];
            u32 minIndex = bvh::minimum(static_cast<u32>(invUnit * (primitiveBBoxes_[index].bmin_[curAxis] - bmin)), NumBins - 1);
            u32 maxIndex = bvh::minimum(static_cast<u32>(invUnit * (primitiveBBoxes_[index].bmax_[curAxis] - bmin)), NumBins - 1);
            BVH_ASSERT(0 <= minIndex && minIndex < NumBins);
            BVH_ASSERT(0 <= maxIndex && maxIndex < NumBins);
            ++minBins[minIndex];
            ++maxBins[maxIndex];
        }

        Vector3 e = extent;
        e[curAxis] = unit[curAxis];
        f32 unitArea = e.halfArea();

        s32 binLeft = 0;
        s32 binRight = NumBins - 1;

        while(minBins[binLeft] <= 0) { ++binLeft; }
        while(maxBins[binRight] <= 0) { --binRight; }
        BVH_ASSERT(0 <= binLeft && binLeft < NumBins);
        BVH_ASSERT(0 <= binRight && binRight < NumBins);

        // SAH with binned AABBs
        s32 n_l = minBins[0];
        s32 n_r = 0;
        for(s32 i = 1; i < binRight; ++i) {
            n_r += maxBins[i];
        }
        for(s32 m = binLeft; m <= binRight; ++m) {
            f32 area_l = m * unitArea;
            f32 area_r = (NumBins - m) * unitArea;
            f32 cost = SAH_KT_ + SAH_KI_ * invArea * (area_l * n_l + area_r * n_r);
            if(cost < bestCost) {
                midBin = m;
                bestCost = cost;
                axis = curAxis;
                bestCentroids = centroids;
            }

            BVH_ASSERT(0 <= m && m < NumBins);
            n_l += minBins[m];
            n_r -= maxBins[m];
        }

        centroids += primitiveIndices_.size();
    } // for(s32 curAxis=0;

    // Sort by the separation value
    f32 separate = unit[axis] * (midBin + 1) + bbox.bmin_[axis];
    u32 mid = start + (numPrimitives >> 1);

    u32 left = start;
    u32 right = end - 1;
    for(;;) {
        while(left < end && bestCentroids[primitiveIndices_[left]] <= separate) {
            ++left;
        }
        while(start <= right && separate < bestCentroids[primitiveIndices_[right]]) {
            --right;
        }
        if(right <= left) {
            mid = left;
            break;
        }
        swap(primitiveIndices_[left], primitiveIndices_[right]);
        ++left;
        --right;
    }

    if(mid <= start || end <= mid) {
        //Cannot split, fallback to splitting at the mid point
        splitMid(axis, num_l, num_r, bbox_l, bbox_r, start, numPrimitives, bbox);
    } else {

        getBBox(bbox_l, start, mid);
        getBBox(bbox_r, mid, end);

        num_l = mid - start;
        num_r = numPrimitives - num_l;
    }
}

HitRecord BinQBVH::intersect(Ray& ray)
{
    vector4_t origin[3];
    vector4_t invDir[3];
    vector4_t tminSSE;
    vector4_t tmaxSSE;
    origin[0] = set1(ray.origin_.x_);
    origin[1] = set1(ray.origin_.y_);
    origin[2] = set1(ray.origin_.z_);

    invDir[0] = set1(ray.invDirection_.x_);
    invDir[1] = set1(ray.invDirection_.y_);
    invDir[2] = set1(ray.invDirection_.z_);

    tminSSE = set1(F32_HITEPSILON);
    tmaxSSE = set1(ray.t_);

    u32 sign[3];
    sign[0] = (0.0f <= ray.direction_[0]) ? 0U : 1U;
    sign[1] = (0.0f <= ray.direction_[1]) ? 0U : 1U;
    sign[2] = (0.0f <= ray.direction_[2]) ? 0U : 1U;

    HitRecord hitRecord;
    hitRecord.t_ = ray.t_;
    hitRecord.primitive_ = HitRecord::Invalid;

    s32 stack = 0;
    u32 nodeStack[MaxDepth << 2];
    nodeStack[0] = 0;
    while(0 <= stack) {
        u32 index = nodeStack[stack];
        const Node& node = nodes_[index];
        BVH_ASSERT(node.leaf_.flags_ == node.joint_.flags_);
        --stack;
        if(node.isLeaf()) {
            u32 primIndex = node.getPrimitiveIndex();
            u32 primEnd = primIndex + node.getNumPrimitives();
            for(u32 i = primIndex; i < primEnd; ++i) {
                f32 t;
                u32 idx = primitiveIndices_[i];
                if(!testRay(t, ray, faces_[idx].p0_, faces_[idx].p1_, faces_[idx].p2_)) {
                    continue;
                }
                if(F32_HITEPSILON < t && t < hitRecord.t_) {
                    ray.t_ = t;
                    hitRecord.t_ = t;
                    hitRecord.primitive_ = idx;
                    tmaxSSE = set1(t);
                }
            } // for(u32 i=primIndex;

        } else {
            // Test ray and quad AABBs
            u32 hit = qbvh::testRayAABB(tminSSE, tmaxSSE, origin, invDir, sign, node.joint_.bbox_);
            u32 split = sign[node.joint_.axis0_] + (sign[node.joint_.axis1_] << 1) + (sign[node.joint_.axis2_] << 2);

            //Traverse according to the ray's direction, so the number of combinations is 2x2x2.
            static const u16 TraverseOrder[] = {
                0x0123U,
                0x2301U,
                0x1023U,
                0x3201U,
                0x0132U,
                0x2301U,
                0x1032U,
                0x3210U,
            };
            u16 order = TraverseOrder[split];
            u32 children = node.joint_.children_;
            for(u32 i = 0; i < 4; ++i) {
                u16 o = order & 0x03U;
                if(hit & (0x01U << o)) {
                    nodeStack[++stack] = children + o;
                }
                order >>= 4;
            }
        }
    } // while(0<=stack){
    return hitRecord;
}

HitRecordSphere BinQBVH::intersect(const Sphere& sphere)
{
    vector4_t position[3];
    position[0] = set1(sphere.center_.x_);
    position[1] = set1(sphere.center_.y_);
    position[2] = set1(sphere.center_.z_);

    vector4_t radius = set1(sphere.radius_);

    HitRecordSphere hitRecord = {};

    s32 stack = 0;
    u32 nodeStack[MaxDepth << 2];
    nodeStack[0] = 0;
    while(0 <= stack) {
        u32 index = nodeStack[stack];
        const Node& node = nodes_[index];
        BVH_ASSERT(node.leaf_.flags_ == node.joint_.flags_);
        --stack;
        if(node.isLeaf()) {
            u32 primIndex = node.getPrimitiveIndex();
            u32 primEnd = primIndex + node.getNumPrimitives();
            for(u32 i = primIndex; i < primEnd; ++i) {
                u32 idx = primitiveIndices_[i];
                f32 t;
                if(!testSphereTriangle(t, sphere, faces_[idx].p0_, faces_[idx].p1_, faces_[idx].p2_)) {
                    continue;
                }
                hitRecord.records_[hitRecord.count_].depth_ = t;
                hitRecord.records_[hitRecord.count_].primitive_ = idx;
                if(HitRecordSphere::MaxHits <= ++hitRecord.count_) {
                    return hitRecord;
                }
            } // for(u32 i=primIndex;

        } else {
            u32 hit = qbvh::testSphereAABB(position, radius, node.joint_.bbox_);
            u32 children = node.joint_.children_;
            for(u32 i = 0; i < 4; ++i) {
                if(hit & (0x01U << i)) {
                    nodeStack[++stack] = children + i;
                }
            }
        }
    } // while(0<=stack){
    return hitRecord;
}

HitRecordCapsule BinQBVH::intersect(const Capsule& capsule)
{
    vector4_t aabb[2][3];
    for(u32 i=0; i<3; ++i){
        vector4_t bmin = set1(capsule.p0_[i]);
        vector4_t bmax = set1(capsule.p1_[i]);
        aabb[0][i] = minimum4(bmin, bmax);
        aabb[1][i] = maximum4(bmin, bmax);
    }
    HitRecordCapsule hitRecord;
    hitRecord.count_ = 0;

    s32 stack = 0;
    u32 nodeStack[MaxDepth << 2];
    nodeStack[0] = 0;
    while(0 <= stack) {
        u32 index = nodeStack[stack];
        const Node& node = nodes_[index];
        BVH_ASSERT(node.leaf_.flags_ == node.joint_.flags_);
        --stack;
        if(node.isLeaf()) {
            u32 primIndex = node.getPrimitiveIndex();
            u32 primEnd = primIndex + node.getNumPrimitives();
            for(u32 i = primIndex; i < primEnd; ++i) {
                f32 t;
                u32 idx = primitiveIndices_[i];
                Vector3 point;
                if(!testCapsuleTriangle(t, point, capsule, faces_[idx].p0_, faces_[idx].p1_, faces_[idx].p2_)) {
                    continue;
                }
                HitRecordCapsule::Record& record = hitRecord.records_[hitRecord.count_];
                record.point_ = point;
                record.depth_ = t;
                record.primitive_ = idx;
                if(HitRecordCapsule::MaxHits<=++hitRecord.count_){
                    return hitRecord;
                }
            } // for(u32 i=primIndex;

        } else {
            u32 hit = qbvh::testAABB(aabb, node.joint_.bbox_);
            u32 children = node.joint_.children_;
            for(u32 i = 0; i < 4; ++i) {
                if(hit & (0x01U << i)) {
                    nodeStack[++stack] = children + i;
                }
            }
        }
    } // while(0<=stack){
    return hitRecord;
}

#if defined(BVH_UE)
#else
#    ifdef _DEBUG
void BinQBVH::print(const char* filename)
{
    FILE* file = BVH_NULL;
#        if defined(_MSC_VER)
    fopen_s(&file, filename, "wb");
#        else
    file = fopen(filename, "wb");
#        endif
    if(BVH_NULL == file) {
        return;
    }

    for(u32 i = 0; i < nodes_.size(); ++i) {
        fprintf(file, "[%d]\n", i);
        if(nodes_[i].isLeaf()) {
            const Leaf& leaf = nodes_[i].leaf_;
            fprintf(file, "  leaf: %d\n", leaf.size_);
        } else {
            const Joint& joint = nodes_[i].joint_;
            const f32* bminx = (f32*)(&joint.bbox_[0][0]);
            const f32* bminy = (f32*)(&joint.bbox_[0][1]);
            const f32* bminz = (f32*)(&joint.bbox_[0][2]);

            const f32* bmaxx = (f32*)(&joint.bbox_[1][0]);
            const f32* bmaxy = (f32*)(&joint.bbox_[1][1]);
            const f32* bmaxz = (f32*)(&joint.bbox_[1][2]);
            fprintf(file, "  joint: %d", joint.children_);

            for(u32 j = 0; j < 4; ++j) {
                fprintf(file, ", bbox:(%f,%f,%f) - (%f,%f,%f)\n", bminx[j], bminy[j], bminz[j], bmaxx[j], bmaxy[j], bmaxz[j]);
            }
        }
    }
    fclose(file);
}
#    endif
#endif
} // namespace bvh
