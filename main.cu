#include "utils.h"

__global__ void randInit(curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    curand_init(RAND_SEED, 0, 0, randState);
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maxX || j >= maxY)
        return;

    int pixelIdx = j * maxX + i;
    curand_init(RAND_SEED + pixelIdx, 0, 0, &randState[pixelIdx]);
}

__device__ vec3 getColor(const ray &r, hittable **world, curandState *localRandState) {
    ray curRay = r;
    vec3 curAttenuation = vec3(1.0f, 1.0f, 1.0f);
    vec3 color = vec3(0.0f, 0.0f, 0.0f);
    for (int depth = 0; depth < RAY_DEPTH; ++depth) {
        hitRecord rec;
        if (!(*world)->hit(curRay, interval(0.001f, FLT_MAX), rec)) {
            color += curAttenuation * ((hittable_list *)(*world))->background;
            return color;
        }

        ray scattered;
        vec3 attenuation = vec3(0.0f, 0.0f, 0.0f);
        vec3 colorFromEmission = rec.matPtr->emitted(rec.u, rec.v, rec.p);
        color += colorFromEmission * curAttenuation;
        if (!rec.matPtr->scatter(curRay, rec, attenuation, scattered, localRandState))
            return color;

        // float scatteringPdf = rec.matPtr->scatteringPdf(r, rec, scattered);
        // float pdf = 1 / (2 * M_PI);
        // curAttenuation *= scatteringPdf / pdf;

        curAttenuation *= attenuation;
        curRay = scattered;
    }

    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= maxX || j >= maxY)
        return;

    int pixelIdx = j * maxX + i;
    curandState localRandState = randState[pixelIdx];
    vec3 color(0, 0, 0);
    for (int s = 0; s < ns; ++s) {
        ray r = (*cam)->getRay(float(i + RND) / float(maxX), float(j + RND) / float(maxY), &localRandState);
        color += getColor(r, world, &localRandState);
    }

    randState[pixelIdx] = localRandState;
    color /= float(ns);
    color[0] = sqrt(color[0]);
    color[1] = sqrt(color[1]);
    color[2] = sqrt(color[2]);
    fb[pixelIdx] = color;
}

__global__ void freeWorld(hittable **dWorld, camera **dCamera) {
    for (hittable *cur = (*dWorld)->nextObject; cur != nullptr; cur = cur->nextObject)
        delete cur;
    delete *dWorld;
    delete *dCamera;
}

__global__ void randomSphere(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0.5f, 0.7f, 1.0f) * 1.0f;
    WORLD->add(new sphere(vec3(0.0f, -10000.0f, -1.0f), 10000.0f, new lambertian(new checkerTexture(0.32f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f)))));

    for (int i = -11; i < 11; ++i)
        for (int j = -11; j < 11; ++j) {
            float chooseMat = RND;
            vec3 center(i + RND, 0.2, j + RND);

            if (chooseMat < 0.8f)
                WORLD->add(new sphere(center, center + vec3(0, RND * 0.5, 0), 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND))));
            else if (chooseMat < 0.95f)
                WORLD->add(new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND)));
            else
                WORLD->add(new sphere(center, 0.2, new dielectric(1.5)));
        }
    WORLD->add(new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5)));
    WORLD->add(new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1))));
    WORLD->add(new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0)));
    WORLD->add(new sphere(vec3(4, 8, 3), 3, new diffuseLight(vec3(1, .9, .6) * 10.0f)));
    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(0, 2, 14);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = 0.01f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void twoSphere(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0.5f, 0.7f, 1.0f);

    auto checker = new checkerTexture(0.8f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f));
    WORLD->add(new sphere(vec3(0, -10, 0), 10, new lambertian(checker)));
    WORLD->add(new sphere(vec3(0, 10, 0), 10, new lambertian(checker)));
    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(13, 2, 3);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = 0.1f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void twoPerlinSphere(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0.5f, 0.7f, 1.0f);

    auto pertext = new noiseTexture(4, randState);
    WORLD->add(new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext)));
    WORLD->add(new sphere(vec3(0, 2, 0), 2, new lambertian(pertext)));
    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(13, 2, 3);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = 0.1f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        20.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void quads(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0.5f, 0.7f, 1.0f);

    // Materials
    auto left_red = new lambertian(vec3(1.0, 0.2, 0.2));
    auto back_green = new lambertian(vec3(0.2, 1.0, 0.2));
    auto right_blue = new lambertian(vec3(0.2, 0.2, 1.0));
    auto upper_orange = new lambertian(vec3(1.0, 0.5, 0.0));
    auto lower_teal = new lambertian(vec3(0.2, 0.8, 0.8));

    WORLD->add(new quad(vec3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), left_red));
    WORLD->add(new quad(vec3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green));
    WORLD->add(new quad(vec3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue));
    WORLD->add(new quad(vec3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange));
    WORLD->add(new quad(vec3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), lower_teal));

    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(0, 0, 9);
    vec3 lookAt(0, 0, 0);
    float focusLen = 10.0f;
    float aperture = .0f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        80.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void simpleLight(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0, 0, 0);

    auto pertext = new noiseTexture(4, randState);
    WORLD->add(new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext)));
    WORLD->add(new sphere(vec3(0, 2, 0), 2, new lambertian(pertext)));

    auto light = new diffuseLight(vec3(4, 4, 4));
    WORLD->add(new sphere(vec3(0, 7, 0), 2, light));
    WORLD->add(new quad(vec3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), light));

    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(26, 3, 6);
    vec3 lookAt(0, 2, 0);
    float focusLen = 10.0f;
    float aperture = .0f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        20.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void cornellBox(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0, 0, 0);

    auto red = new lambertian(vec3(.65, .05, .05));
    auto white = new lambertian(vec3(.73, .73, .73));
    auto green = new lambertian(vec3(.12, .45, .15));
    auto light = new diffuseLight(vec3(15, 15, 15));

    WORLD->add(new quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
    WORLD->add(new quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
    WORLD->add(new quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light));
    WORLD->add(new quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
    WORLD->add(new quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white));
    WORLD->add(new quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));

    auto aluminum = new metal(vec3(0.8, 0.85, 0.88), 0.0);
    hittable *box1 = box(vec3(0, 0, 0), vec3(165, 330, 165), aluminum);
    box1 = new rotateY(box1, 15);
    box1 = new translate(box1, vec3(265, 0, 295));
    WORLD->add(box1);

    hittable *box2 = box(vec3(0, 0, 0), vec3(165, 165, 165), white);
    box2 = new rotateY(box2, -18);
    box2 = new translate(box2, vec3(130, 0, 65));
    WORLD->add(box2);

    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(278, 278, -800);
    vec3 lookAt(278, 278, 0);
    float focusLen = 10.0f;
    float aperture = .0f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        40.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void cornellSmoke(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0, 0, 0);

    auto red = new lambertian(vec3(.65, .05, .05));
    auto white = new lambertian(vec3(.73, .73, .73));
    auto green = new lambertian(vec3(.12, .45, .15));
    auto light = new diffuseLight(vec3(7, 7, 7));

    WORLD->add(new quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green));
    WORLD->add(new quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red));
    WORLD->add(new quad(vec3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), light));
    WORLD->add(new quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white));
    WORLD->add(new quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white));
    WORLD->add(new quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white));

    hittable *box1 = box(vec3(0, 0, 0), vec3(165, 330, 165), white);
    box1 = new rotateY(box1, 15);
    box1 = new translate(box1, vec3(265, 0, 295));

    hittable *box2 = box(vec3(0, 0, 0), vec3(165, 165, 165), white);
    box2 = new rotateY(box2, -18);
    box2 = new translate(box2, vec3(130, 0, 65));

    WORLD->add(new constant_medium(box1, 0.01, vec3(0, 0, 0), randState));
    WORLD->add(new constant_medium(box2, 0.01, vec3(1, 1, 1), randState));

    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(278, 278, -800);
    vec3 lookAt(278, 278, 0);
    float focusLen = 10.0f;
    float aperture = .0f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        40.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

__global__ void finalScene(hittable **dWorld, camera **dCamera, int nx, int ny, curandState *randState) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    curandState localRandState = *randState;
    *dWorld = new hittable_list();
    WORLD->background = vec3(0, 0, 0);

    int boxes_per_side = 20;
    auto ground = new lambertian(vec3(0.48, 0.83, 0.53));
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = RND * 100 + 1;
            auto z1 = z0 + w;

            WORLD->add(box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground));
        }
    }

    auto light = new diffuseLight(vec3(7, 7, 7));
    WORLD->add(new quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light));

    auto center1 = vec3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto sphere_material = new lambertian(vec3(0.7, 0.3, 0.1));
    WORLD->add(new sphere(center1, center2, 50, sphere_material));

    WORLD->add(new sphere(vec3(260, 150, 45), 50, new dielectric(1.5)));
    WORLD->add(new sphere(
        vec3(0, 150, 145), 50, new metal(vec3(0.8, 0.8, 0.9), 1.0)
    ));

    auto boundary = new sphere(vec3(360, 150, 145), 70, new dielectric(1.5));
    WORLD->add(boundary);
    WORLD->add(new constant_medium(boundary, 0.2, vec3(0.2, 0.4, 0.9), randState));
    boundary = new sphere(vec3(0, 0, 0), 5000, new dielectric(1.5));
    WORLD->add(new constant_medium(boundary, .0001, vec3(1, 1, 1), randState));

    // auto emat = new lambertian(new image_texture("earthmap.jpg"));
    // WORLD->add(new sphere(vec3(400,200,400), 100, emat));
    // auto pertext = new noise_texture(0.1);
    // WORLD->add(new sphere(vec3(220,280,300), 80, new lambertian(pertext)));

    auto balls = new hittable_list();
    auto white = new lambertian(vec3(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        balls->add(new sphere(vec3(RND, RND, RND) * 165, 10, white));
    }
    WORLD->add(new translate(new rotateY(balls, 15), vec3(-100, 270, 395)));

    *randState = localRandState;
    bvhNode::buildFromList(dWorld);

    vec3 lookFrom(478, 278, -600);
    vec3 lookAt(278, 278, 0);
    float focusLen = 10.0f;
    float aperture = .0f;
    *dCamera = new camera(
        lookFrom,
        lookAt,
        vec3(0, 1, 0),
        40.0,
        float(nx) / float(ny),
        aperture,
        focusLen
    );
}

int main(int argc, char const *argv[]) {
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int nx = IMAGE_WIDTH;
    int ny = IMAGE_HEIGHT;
    int ns = SAMPLE_PER_PIXEL;
    int tx = 32;
    int ty = 32;
    size_t stackSize = 1 << 12;

    int seed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    checkCudaErrors(cudaMalloc((void **)&dSeed, sizeof(int)));
    checkCudaErrors(cudaMemcpy(dSeed, &seed, sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
    std::cerr << "CUDA Stack Size Limit: " << stackSize << " bytes\n";

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int pixelNum = nx * ny;
    size_t fbSize = pixelNum * sizeof(vec3);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fbSize));

    curandState *dRandState;
    checkCudaErrors(cudaMalloc((void **)&dRandState, pixelNum * sizeof(curandState)));
    curandState *dRandState_;
    checkCudaErrors(cudaMalloc((void **)&dRandState_, 1 * sizeof(curandState)));

    randInit << <1, 1 >> > (dRandState_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable **dWorld;
    checkCudaErrors(cudaMalloc((void **)&dWorld, sizeof(hittable *)));
    camera **dCamera;
    checkCudaErrors(cudaMalloc((void **)&dCamera, sizeof(camera *)));
    switch (7) {
    case 0:
        randomSphere << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);;
        break;
    case 1:
        twoSphere << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 2:
        twoPerlinSphere << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 3:
        quads << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 4:
        simpleLight << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 5:
        cornellBox << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 6:
        cornellSmoke << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    case 7:
        finalScene << <1, 1 >> > (dWorld, dCamera, nx, ny, dRandState_);
        break;
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "World created.\n";

    auto start = std::chrono::high_resolution_clock::now();

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    renderInit << <blocks, threads >> > (nx, ny, dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, nx, ny, ns, dCamera, dWorld, dRandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "\ntook " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0f << " s.\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixelIdx = j * nx + i;
            int ir = (int)(fmin(256, 255.99 * fb[pixelIdx].r()));
            int ig = (int)(fmin(256, 255.99 * fb[pixelIdx].g()));
            int ib = (int)(fmin(256, 255.99 * fb[pixelIdx].b()));
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    freeWorld << <1, 1 >> > (dWorld, dCamera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dCamera));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(dRandState));
    checkCudaErrors(cudaFree(dRandState_));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaDeviceReset());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}
