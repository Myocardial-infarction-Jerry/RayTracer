#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center1(cen), radius(r), mat_ptr(m), is_moving(false) {
        vec3 rvec = vec3(r, r, r);
        bbox = aabb(cen - rvec, cen + rvec);
    };
    __device__ sphere(point3 _center1, point3 _center2, float r, material *m) : center1(_center1), radius(r), mat_ptr(m), is_moving(true) {
        center_vec = _center2 - _center1;
        vec3 rvec = vec3(r, r, r);
        aabb box1(_center1 - rvec, _center1 + rvec);
        aabb box2(_center2 - rvec, _center2 + rvec);
        bbox = aabb(box1, box2);
    };

    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    __device__  aabb bounding_box() const { return bbox; }
    __device__ point3 sphere_center(double time) const { return center1 + time * center_vec; }

    vec3 center1;
    bool is_moving;
    vec3 center_vec;
    float radius;
    material *mat_ptr;
    aabb bbox;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    point3 center = is_moving ? sphere_center(r.time()) : center1;
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif
