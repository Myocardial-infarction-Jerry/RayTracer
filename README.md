# RayTracer

Base on https://github.com/RayTracing/raytracing.github.io

Build on WSL

## Ray intersect Sphere

$$
\begin{aligned}
Ray&:\qquad R(t)=\vec o+\vec dt\\
Sphere&:\qquad S(\vec p)=||\vec p-\vec c||-r=0
\end{aligned}
$$

$$
S(R(t))=||\vec o-\vec c+\vec dt||-r=0\\
||\vec o-\vec c+\vec dt||=r\\
(\vec o-\vec c+\vec dt)^2=r^2\\
t^2||\vec d||^2+2t\cdot(\vec d\cdot(\vec o-\vec c))+||\vec o-\vec c||^2-r^2=0
$$

For quadratic equations

$$
ax^2+bx+c=0\\
\Delta=b^2-4ac\\
\begin{aligned}
x&=\frac{-b\pm\sqrt \Delta}{2a}\\
&=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\
&=\frac{-2b'\pm\sqrt{4b'^2-4ac}}{2a}\qquad (b=2b')\\
&=\frac{-b'\pm\sqrt{b'^2-ac}}{a}
\end{aligned}
$$

Substitution can be obtained

$$
a=||\vec d||^2,\quad b'=\vec d\cdot(\vec o-\vec c),\quad c=||\vec o-\vec c||^2-r^2\\
t_i=\frac{-b'-\sqrt{b'^2-ac}}{a}
$$

if $t_i<0$, then the origin of the ray is ***inside the sphere*** or the sphere is ***on the negative side of the ray***.

## Ray Reflection & Refraction

We set normal vector $||\vec n||=1$, therefore

$$
\vec n\cdot\vec I=||\vec n||\cdot||\vec I||\cdot\cos\theta=||\vec I||\cdot\cos\theta\\
\vec I'=\vec I-2\vec I(\vec I\cdot\vec n)=\vec I(1-2(\vec I\cdot\vec n))
$$

Occasionally, $\vec I=\vec d$, therefore

$$
\vec d'=\vec d-2\vec d(\vec d\cdot\vec n)=\vec d(1-2(\vec d\cdot \vec n))
$$
