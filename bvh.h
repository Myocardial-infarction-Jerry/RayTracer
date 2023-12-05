#ifndef BVH_H
#define BVH_H

#include "utils.h"
#include "hittable.h"
#include "hittable_list.h"

class bvhNode :public hittable {
public:
    __device__ bvhNode() :start(nullptr), end(nullptr), left(nullptr), right(nullptr) {}
    __device__ bvhNode(hittable *_start, hittable *_end) : start(_start), end(_end), left(nullptr), right(nullptr) {}
    __device__ bvhNode(const hittable_list &objList) : start(objList.nextObject), end(nullptr), left(nullptr), right(nullptr) {
        int objectNum = 0;
        for (hittable *cur = start; cur != end; cur = cur->nextObject) ++objectNum;
        hittable **list = new hittable * [objectNum * 4];
        int size = 0;

        curandState localRandState;
        curand_init(RAND_SEED, 0, 0, &localRandState);

        list[size++] = this;
        for (int i = 0; i < size; ++i) {
            bvhNode *cur = (bvhNode *)(list[i]);
            if (cur->start == cur->end)
                continue;
            int axis = int(RND * 3);
            auto comparator = (axis == 0) ? boxXCompare : (axis == 1) ? boxYCompare : boxZCompare;
            hittable *mid = bvhSort(cur->start, comparator);

            if (cur->start->nextObject == mid) {
                cur->left = cur->start;
                cur->left->nextObject = nullptr;
            }
            else
                cur->left = list[size++] = new bvhNode(cur->start, mid);

            if (mid->nextObject == cur->end) {
                cur->right = mid;
                cur->right->nextObject = nullptr;
            }
            else
                cur->right = list[size++] = new bvhNode(mid, cur->end);
        }

        for (int i = size - 1; i >= 0; --i) {
            bvhNode *cur = (bvhNode *)(list[i]);
            cur->bbox = aabb(left->boundingBox(), right->boundingBox());
        }

        delete[] list;
    }

    __device__ virtual bool hit(const ray &r, const interval &rayT, hitRecord &rec) const override {
        if (!bbox.hit(r, rayT))
            return false;

        bool hitLeft = left->hit(r, rayT, rec);
        bool hitRight = right->hit(r, interval(rayT.min, hitLeft ? rec.T : rayT.max), rec);

        return hitLeft || hitRight;
    }

    __device__ virtual aabb boundingBox() const override { return bbox; }

    __device__ inline static bool boxCompare(hittable *a, hittable *b, int axisIdx) { return a->boundingBox().axis(axisIdx).min < b->boundingBox().axis(axisIdx).min; }
    __device__ inline static bool boxXCompare(hittable *a, hittable *b) { return boxCompare(a, b, 0); }
    __device__ inline static bool boxYCompare(hittable *a, hittable *b) { return boxCompare(a, b, 1); }
    __device__ inline static bool boxZCompare(hittable *a, hittable *b) { return boxCompare(a, b, 2); }

    __device__ static hittable *bvhSort(hittable *head, bool(*comparator)(hittable *, hittable *)) {
        if (head == nullptr || head->nextObject == nullptr)
            return head;

        hittable *dummy = new bvhNode();
        dummy->nextObject = head;

        int length = 0;
        while (head != nullptr) {
            length++;
            head = head->nextObject;
        }

        for (int step = 1; step < length; step <<= 1) {
            hittable *pre = dummy;
            hittable *cur = dummy->nextObject;
            while (cur != nullptr) {
                hittable *left = cur;
                hittable *right = split(left, step);
                cur = split(right, step);
                pre = merge(left, right, pre, comparator);
            }
        }

        return dummy->nextObject;
    }

    __device__ static hittable *split(hittable *node, int step) {
        if (node == nullptr)
            return nullptr;

        for (int i = 1; node->nextObject != nullptr && i < step; i++) {
            node = node->nextObject;
        }
        hittable *right = node->nextObject;
        node->nextObject = nullptr;

        return right;
    }

    __device__ static hittable *merge(hittable *left, hittable *right, hittable *pre, bool(*comparator)(hittable *, hittable *)) {
        hittable *cur = pre;
        while (left != nullptr && right != nullptr) {
            if (comparator(left, right)) {
                cur->nextObject = left;
                left = left->nextObject;
            }
            else {
                cur->nextObject = right;
                right = right->nextObject;
            }
            cur = cur->nextObject;
        }
        if (left == nullptr) cur->nextObject = right;
        if (right == nullptr) cur->nextObject = left;
        while (cur->nextObject != nullptr) cur = cur->nextObject;
        return cur;
    }

    // private:
    hittable *start, *end;
    hittable *left, *right;
    aabb bbox;
};

#endif