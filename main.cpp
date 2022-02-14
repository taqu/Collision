#include "BinQBVH.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include "sibenik/sibenik.bin"
#include "conference/conference.bin"

using namespace bvh;

namespace
{
    u32 width = 640;
    u32 height = 480;

    Vector3 cameraPosition;
    Vector3 cameraForward;
    Vector3 cameraUp;
    Vector3 cameraRight;
    f32 iw;
    f32 ih;

    Vector3 sx;
    Vector3 sy;

    RGB* rgb = NULL;

    void initScene(const Vector3 position, const Vector3 direction)
    {
        cameraPosition = position;
        cameraForward = normalize(direction);
        cameraUp = {0.0f, 1.0f, 0.0f};
        cameraRight = normalize(cross(cameraUp, cameraForward));
        f32 aspect = static_cast<f32>(width)/height;

        f32 fovy = ::tanf(0.5f * 60.0f/180.0f*static_cast<f32>(BVH_PI));
        f32 dx = fovy * aspect;
        f32 dy = fovy;

        iw = 1.0f/(width-1);
        ih = 1.0f/(height-1);
        sx = dx * cameraRight;
        sy = dy * cameraUp;

        rgb = (RGB*)BVH_MALLOC(sizeof(RGB)*width*height);
    }

    void termScene()
    {
        BVH_FREE(rgb);
    }
}

void test(
    u32& depth,
    long long& buildTime,
    long long& renderTime,
    f64& raysPerSecond,
    f64& rayAndSpheresPerSecond,
    u32 numVertices,
    const Vector3* vertices,
    const char* filename)
{
    BinQBVH bvh;
    u32 numTriangles = numVertices/3;
    bvh.resize(numTriangles);
    for(u32 i=0; i<numTriangles; ++i){
        u32 index = i*3;
        Face& face = bvh.face(i);
        face.p0_ = vertices[index+0];
        face.p1_ = vertices[index+1];
        face.p2_ = vertices[index+2];
    }

    std::chrono::high_resolution_clock::time_point start;
    start = std::chrono::high_resolution_clock::now();
    bvh.build();
    buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    depth = bvh.getDepth();

    Ray ray;
    ray.origin_ = cameraPosition;
    ray.t_ = 1.0e32f;

    start = std::chrono::high_resolution_clock::now();
    for(u32 i=0; i<height; ++i){
        Vector3 vy = sy * (1.0f - 2.0f*i*ih);
        for(u32 j=0; j<width; ++j){
            Vector3 vx = sx * (2.0f*j*iw - 1.0f);

            ray.direction_ = normalize(cameraForward+vx+vy);
            ray.invertDirection();
            ray.t_ = 1.0e32f;
            RGB& pixel = rgb[i*width+j];
            HitRecord hitRecord = bvh.intersect(ray);
            if(HitRecord::Invalid != hitRecord.primitive_){
                const Face& face = bvh.face(hitRecord.primitive_);
                Vector3 d0 = face.p1_ - face.p0_;
                Vector3 d1 = face.p2_ - face.p0_;
                Vector3 normal = normalize(cross(d0, d1));
                normal.y_ = absolute(normal.y_);
                pixel.r_ = pixel.g_ = pixel.b_ = maximum(normal.y_, 0.0f);
            }else{
                pixel.r_ = pixel.g_ = pixel.b_ = 0;
            }
        }
    }
    renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    raysPerSecond = static_cast<f64>(height*width*1000)/renderTime;
    printImage(filename, rgb, width, height);

    start = std::chrono::high_resolution_clock::now();
    for(u32 i=0; i<height; ++i){
        Vector3 vy = sy * (1.0f - 2.0f*i*ih);
        for(u32 j=0; j<width; ++j){
            Vector3 vx = sx * (2.0f*j*iw - 1.0f);

            ray.direction_ = normalize(cameraForward+vx+vy);
            ray.invertDirection();
            ray.t_ = 1.0e32f;
            RGB& pixel = rgb[i*width+j];
            HitRecord hitRecord = bvh.intersect(ray);
            if(HitRecord::Invalid != hitRecord.primitive_){
                Sphere sphere;
                sphere.center_ = hitRecord.t_*ray.direction_ + ray.origin_;
                sphere.radius_ = 1.0e-4f;
                pixel.r_ = pixel.g_ = pixel.b_ = 0;
                HitRecordSet hitRecordSet = bvh.intersect(sphere);
                if(0<hitRecordSet.count_){
                    pixel.r_ = pixel.g_ = pixel.b_ = 1.0f;
                }
            }else{
                pixel.r_ = pixel.g_ = pixel.b_ = 0;
            }
        }
    }
    long long raySphereTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    rayAndSpheresPerSecond = static_cast<f64>(height*width*1000)/raySphereTime;
    {
        char buffer[256];
        sprintf(buffer, "sphere_%s", filename);
        printImage(buffer, rgb, width, height);
    }
    printf("%s is done.\n", filename);
}

void test(
    FILE* file,
    u32 numVertices,
    const Vector3* vertices,
    const char* name,
    const char* filename)
{
    u32 depth=0;
    long long buildTime=0;
    long long renderTime=0;
    f64 raysPerSecond=0.0;
    f64 rayAndSpheresPerSecond=0.0;
    test(depth, buildTime, renderTime, raysPerSecond, rayAndSpheresPerSecond, numVertices, vertices, filename);
    fprintf(file, "%s depth: %d, build: %lld, render: %lld, rays: %f, spheres: %f\r\n", name, depth, buildTime, renderTime, raysPerSecond, rayAndSpheresPerSecond);
}

int main(void)
{
#if 1
    FILE* file = NULL;
#if defined(_MSC_VER)
    fopen_s(&file, "statistics.txt", "wb");
#else
    file = fopen("statistics.txt", "wb");
#endif
    if(NULL == file){
        return 0;
    }

    const u32 SibenikNumFaces = sizeof(sibenik_indices)/sizeof(sibenik_indices[0])/3;
    Vector3* vertices = (Vector3*)BVH_MALLOC(sizeof(Vector3) * SibenikNumFaces * 3);
    for(u32 i=0; i<SibenikNumFaces; ++i){
        u32 index = i*3;
        u32 idx0 = sibenik_indices[index+0]*3;
        u32 idx1 = sibenik_indices[index+1]*3;
        u32 idx2 = sibenik_indices[index+2]*3;
        vertices[index + 0] = {sibenik_vertices[idx0+0], sibenik_vertices[idx0+1], sibenik_vertices[idx0+2]};
        vertices[index + 1] = {sibenik_vertices[idx1+0], sibenik_vertices[idx1+1], sibenik_vertices[idx1+2]};
        vertices[index + 2] = {sibenik_vertices[idx2+0], sibenik_vertices[idx2+1], sibenik_vertices[idx2+2]};
    }
    initScene({-15.0f, -5.0f, 0.0f}, {1.0f, -0.2f, 0.0f});

    fprintf(file, "sibenik num primitives: %d\r\n", SibenikNumFaces);
    test(file, SibenikNumFaces*3, vertices, "binqbvh", "sibenik_binqbvh.ppm");

    termScene();

    BVH_FREE(vertices);


    const int ConferenceNumFaces = sizeof(conference_indices)/sizeof(conference_indices[0])/3;
    vertices = (Vector3*)BVH_MALLOC(sizeof(Vector3) * ConferenceNumFaces * 3);
    for(u32 i=0; i<ConferenceNumFaces; ++i){
        u32 index = i*3;
        u32 idx0 = conference_indices[index+0]*3;
        u32 idx1 = conference_indices[index+1]*3;
        u32 idx2 = conference_indices[index+2]*3;

        vertices[index + 0] = {conference_vertices[idx0 + 0], conference_vertices[idx0 + 1], conference_vertices[idx0 + 2]};
        vertices[index + 1] = {conference_vertices[idx1 + 0], conference_vertices[idx1 + 1], conference_vertices[idx1 + 2]};
        vertices[index + 2] = {conference_vertices[idx2 + 0], conference_vertices[idx2 + 1], conference_vertices[idx2 + 2]};
    }
    initScene({1610.0f, 356.0f, -322.0f}, {-100.0f, 0.0f, 0.0f});

    fprintf(file, "conference num primitives: %d\r\n", ConferenceNumFaces);
    test(file, ConferenceNumFaces*3, vertices, "binqbvh", "conference_binqbvh.ppm");
    termScene();

    BVH_FREE(vertices);
    fclose(file);
    #endif
    return 0;
}
