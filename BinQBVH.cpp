﻿/**
@file BinQBVH.cpp
@author t-sakai
@date 2018/01/22 create
*/
#include "BinQBVH.h"
#if defined(BVH_UE)
#else
#include <cmath>
#include <cstdio>
#endif

#    if defined(BVH_NEON)
bvh_int32_t movemask(uint32x4_t x)
{
    bvh_uint32_t mask[4];
    vst1q_u32(mask, x);
    return ((mask[3] & 0x01U) << 3) | ((mask[2] & 0x01U) << 2) | ((mask[1] & 0x01U) << 1) | ((mask[0] & 0x01U) << 0);
}
#    endif

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

#	if defined(BVH_UE)
#else
//--- RGB
//--------------------------------------------------------------
void printImage(const char* filename, RGB* rgb, u32 width, u32 height)
{
    BVH_ASSERT(BVH_NULL != filename);
    BVH_ASSERT(BVH_NULL != rgb);

    FILE* file = BVH_NULL;
#if defined(_MSC_VER)
    fopen_s(&file, filename, "wb");
#else
    file = fopen(filename, "wb");
#endif
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

f32 Vector3::length() const
{
    return BVH_SQRT(x_ * x_ + y_ * y_ + z_ * z_);
}

f32 Vector3::halfArea() const
{
    return x_ * y_ + y_ * z_ + z_ * x_;
}

Vector3& Vector3::operator*=(f32 a)
{
    x_ *= a;
    y_ *= a;
    z_ *= a;
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

#if 0
    bool AABB::testRay(f32& tmin, f32& tmax, const Ray& ray) const
    {
        tmin = 0.0f;
        tmax = ray.t_;

        for(s32 i=0; i<3; ++i){
            if(absolute(ray.direction_[i])<F32_HITEPSILON){
                //線分とスラブが平行で、原点がスラブの中にない
                if(ray.origin_[i]<bmin_[i] || bmax_[i]<ray.origin_[i]){
                    return false;
                }

            }else{
                f32 invD = ray.invDirection_[i];
                f32 t1 = (bmin_[i] - ray.origin_[i]) * invD;
                f32 t2 = (bmax_[i] - ray.origin_[i]) * invD;

                if(t1>t2){
                    if(t2>tmin) tmin = t2;
                    if(t1<tmax) tmax = t1;
                }else{
                    if(t1>tmin) tmin = t1;
                    if(t2<tmax) tmax = t2;
                }

                if(tmin > tmax){
                    return false;
                }
                if(tmax < 0.0f){
                    return false;
                }
            }
        }
        return true;
    }
#endif

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
        // Parallel face
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

void closestPointSegment(Vector3& closestPoint, const Vector3& start, const Vector3& end, const Vector3& point)
{
    Vector3 direction = end - start;
    f32 t = dot(point - start, direction);
    if(t < -F32_EPSILON) {
        closestPoint = start;
    } else {
        f32 denom = dot(direction, direction);
        if(denom <= t) {
            closestPoint = end;
        } else {
            t = t / denom;
            closestPoint = start + (direction * t);
        }
    }
}

f32 testSphereCapsule(const Sphere& sphere, const Capsule& capsule)
{
    f32 d2 = sqrDistancePointSegment(capsule.p0_, capsule.p1_, sphere.center_);
    return sphere.radius_ - BVH_SQRT(d2);
}

bool testSphereTriangle(HitRecordSet::Record& record, const Sphere& sphere, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
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
    f32 aa = dot(A, A);
    f32 bb = dot(B, B);
    f32 cc = dot(C, C);
    f32 ab = dot(A, B);
    f32 ac = dot(A, C);
    f32 bc = dot(B, C);
    bool sep1 = (rr < aa) && (aa < ab) && (aa < ac);
    bool sep2 = (rr < bb) && (bb < ab) && (bb < bc);
    bool sep3 = (rr < cc) && (cc < ac) && (cc < bc);
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
    bool sep4 = (e0 * e0 * rr < dot(Q0, Q0)) && (0.0f < dot(Q0, QC));
    bool sep5 = (e1 * e1 * rr < dot(Q1, Q1)) && (0.0f < dot(Q1, QA));
    bool sep6 = (e2 * e2 * rr < dot(Q2, Q2)) && (0.0f < dot(Q2, QB));
    if(sep0 || sep1 || sep2 || sep3 || sep4 || sep5 || sep6){
        Vector3 normal = normalize(V);
        f32 distance = dot(sphere.center_, normal) + dot(normal, p0);
        record.direction_ = normal;
        record.depth_ = -distance;
        return true;
    }else{
        return false;
    }
}

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
        const Vector4 bbox[2][3])
    {
        for(s32 i = 0; i < 3; ++i) {
            vector4_t b0 = load(reinterpret_cast<const f32*>(&bbox[sign[i]][i]));
            tmin = maximum4(
                tmin,
                mul(sub(b0, origin[i]), invDir[i]));

            vector4_t b1 = load(reinterpret_cast<const f32*>(&bbox[1 - sign[i]][i]));
            tmax = minimum4(
                tmax,
                mul(sub(b1, origin[i]), invDir[i]));
        }
        return movemask(cmpge(tmax, tmin));
    }

    // Collide aabb vs aabb
    s32 testAABB(const vector4_t bbox0[2][3], const vector4_t bbox1[2][3])
    {
        union Mask
        {
            u32 u_;
            f32 f_;
        };
        Mask mask;
        mask.u_ = 0xFFFFFFFFU;

#if defined(BVH_NEON) || defined(BVH_SOFT)
		uvector4_t t = set1_u32(mask.u_);
#else
		vector4_t t = set1(mask.f_);
#endif
        for(u32 i = 0; i < 3; ++i) {
            t = and4(t, cmple(bbox0[0][i], bbox1[1][i]));
            t = and4(t, cmple(bbox1[0][i], bbox0[1][i]));
        }
        return movemask(t);
    }

    // Collide sphre vs aabb
    s32 testSphereAABB(const vector4_t position[3], const vector4_t radius, const Vector4 bbox[2][3])
    {
        union Mask
        {
            u32 u_;
            f32 f_;
        };
        Mask mask;
        mask.u_ = 0xFFFFFFFFU;

        vector4_t tbbox[2][3];
        for(u32 i = 0; i < 3; ++i) {
            vector4_t b0 = load(reinterpret_cast<const f32*>(&bbox[0][i]));
            tbbox[0][i] = sub(b0, radius);
            vector4_t b1 = load(reinterpret_cast<const f32*>(&bbox[1][i]));
            tbbox[1][i] = add(b1, radius);
        }
#if defined(BVH_NEON) || defined(BVH_SOFT)
		uvector4_t t = set1_u32(mask.u_);
#else
		vector4_t t = set1(mask.f_);
#endif
        for(u32 i = 0; i < 3; ++i) {
            t = and4(t, cmple(position[i], tbbox[1][i]));
            t = and4(t, cmple(tbbox[0][i], position[i]));
        }
        return movemask(t);
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
            nodes_.reserve(nodes_.capacity() << 1);
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

void BinQBVH::split(u8& axis, u32& num_l, u32& num_r, AABB& bbox_l, AABB& bbox_r, f32 invArea, u32 start, u32 numPrimitives, const AABB& bbox)
{
    u32 end = start + numPrimitives;
    u32 mid = start + (numPrimitives >> 1);

    f32 area_l, area_r;
    f32 bestCost = bvh_limits_max;

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

    Vector3 extent = bbox.extent();
    Vector3 unit = extent * (1.0f / NumBins);
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

    f32 separate = unit[axis] * (midBin + 1) + bbox.bmin_[axis];
    u32 mid = start + (numPrimitives >> 1);

#if 1
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
#else
    PrimitivePolicy::sort(numPrimitives, &primitiveIndices_[start], bestCentroids);
    for(s32 i = start; i < end; ++i) {
        if(separate < bestCentroids[primitiveIndices_[i]]) {
            mid = i;
            break;
        }
    }
#endif

    if(mid <= start || end <= mid) {
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

    s32 raySign[3];
    raySign[0] = (0.0f <= ray.direction_[0]) ? 0 : 1;
    raySign[1] = (0.0f <= ray.direction_[1]) ? 0 : 1;
    raySign[2] = (0.0f <= ray.direction_[2]) ? 0 : 1;

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
            u32 hit = qbvh::testRayAABB(tminSSE, tmaxSSE, origin, invDir, raySign, node.joint_.bbox_);
            u32 split = raySign[node.joint_.axis0_] + (raySign[node.joint_.axis1_] << 1) + (raySign[node.joint_.axis2_] << 2);

            //それぞれの分割で反転するか. 2x2x2
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

HitRecordSet BinQBVH::intersect(const Sphere& sphere)
{
    vector4_t position[3];
    position[0] = set1(sphere.center_.x_);
    position[1] = set1(sphere.center_.y_);
    position[2] = set1(sphere.center_.z_);

    vector4_t radius = set1(sphere.radius_);

    HitRecordSet hitRecord = {};

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
                if(!testSphereTriangle(hitRecord.records_[hitRecord.count_], sphere, faces_[idx].p0_, faces_[idx].p1_, faces_[idx].p2_)) {
                    continue;
                }
                hitRecord.records_[hitRecord.count_].primitive_ = idx;
                if(HitRecordSet::MaxHits<=++hitRecord.count_){
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

#	if defined(BVH_UE)
#else
#ifdef _DEBUG
void BinQBVH::print(const char* filename)
{
    FILE* file = BVH_NULL;
#    if defined(_MSC_VER)
    fopen_s(&file, filename, "wb");
#    else
    file = fopen(filename, "wb");
#    endif
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
#endif
#endif
} // namespace bvh
