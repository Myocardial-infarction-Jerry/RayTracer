#include "fragment.h"

#define EPSILON 0.000001f

Fragment::Fragment() {}

hitRecord Fragment::hit(const Ray &ray) const {
    hitRecord record;
    Vec3 n = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);

    if (fabs(ray.direction.dot(n)) < EPSILON) {
        record.hit = false;
        return record;
    }

    float t = (vertices[0] - ray.origin).dot(n) / ray.direction.dot(n);

    if (t < 0) {
        record.hit = false;
        return record;
    }

    Vec3 p = ray.origin + t * ray.direction;

    // Calculate barycentric coordinates
    float u = ((vertices[1] - vertices[0]).cross(p - vertices[0])).dot(n);
    float v = ((vertices[2] - vertices[1]).cross(p - vertices[1])).dot(n);
    float w = ((vertices[0] - vertices[2]).cross(p - vertices[2])).dot(n);

    // Check if point is inside the triangle
    if ((u >= 0 && v >= 0 && w >= 0) || (u <= 0 && v <= 0 && w <= 0)) {
        record.hit = true;
        record.t = t;
        record.p = p;
        record.normal = n;
        record.material = material;
    }
    else {
        record.hit = false;
    }

    return record;
}