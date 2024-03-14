class Interval {
public:
    float mmin, mmax;

    Interval();
    Interval(float val);
    Interval(float min, float max);

    bool contain(float val);

    Interval operator+(const Interval &other);
    Interval &operator+=(const Interval &other);
};