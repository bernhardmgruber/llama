#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>
#include <tiny_obj_loader.h>

namespace
{
    // clang-format off
    struct X {};
    struct Y {};
    struct Z {};
    struct Vertex0 {};
    struct Edge1 {};
    struct Edge2 {};
    // clang-format on

    using Vec = llama::DS<llama::DE<X, float>, llama::DE<Y, float>, llama::DE<Z, float>>;
    using PrepTriangle = llama::DS<llama::DE<Vertex0, Vec>, llama::DE<Edge1, Vec>, llama::DE<Edge2, Vec>>;

    using ArrayDomain = llama::ArrayDomain<1>;
    // using Mapping = llama::mapping::AoS<ArrayDomain, PrepTriangle>;
    using Mapping = llama::mapping::SoA<ArrayDomain, PrepTriangle, std::true_type>;
    // using Mapping = llama::mapping::AoSoA<ArrayDomain, PrepTriangle, 8>;
    using TriangleView = decltype(llama::allocView(Mapping{llama::ArrayDomain{0}}));

    template <typename F>
    struct Vector
    {
        auto operator[](std::size_t index) -> F&
        {
            return values[index];
        }

        auto operator[](std::size_t index) const -> F
        {
            return values[index];
        }

        auto operator-() const -> Vector
        {
            return {-values[0], -values[1], -values[2]};
        }

        auto operator+=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] += v.values[i];
            return *this;
        }

        auto operator-=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] -= v.values[i];
            return *this;
        }

        template <typename Scalar>
        auto operator*=(Scalar scalar) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] *= scalar;
            return *this;
        }

        auto lengthSqr() const
        {
            F r = 0;
            for (int i = 0; i < 3; i++)
                r += values[i] * values[i];
            return r;
        }

        auto length() const
        {
            return std::sqrt(lengthSqr());
        }

        void normalize()
        {
            const auto l = length();
            for (int i = 0; i < 3; i++)
                values[i] /= l;
        }

        auto normalized() const -> Vector
        {
            auto r = *this;
            r.normalize();
            return r;
        }

        friend auto operator+(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r += b;
            return r;
        }

        friend auto operator-(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r -= b;
            return r;
        }

        friend auto operator*(const Vector& v, F scalar) -> Vector
        {
            auto r = v;
            r *= scalar;
            return r;
        }

        friend auto operator*(F scalar, const Vector& v) -> Vector
        {
            return v * scalar;
        }

        friend auto operator>>(std::istream& is, Vector& v) -> std::istream&
        {
            for (int i = 0; i < 3; i++)
                is >> v[i];
            return is;
        }

        friend auto operator<<(std::ostream& os, const Vector& v) -> std::ostream&
        {
            for (int i = 0; i < 3; i++)
                os << v[i] << " ";
            return os;
        }

        std::array<F, 3> values = {{0, 0, 0}};
    };

    template <typename F>
    inline auto dot(const Vector<F>& a, const Vector<F>& b) -> F
    {
        F r = 0;
        for (int i = 0; i < 3; i++)
            r += a[i] * b[i];
        return r;
    }

    template <typename F>
    inline auto cross(const Vector<F>& a, const Vector<F>& b) -> Vector<F>
    {
        Vector<F> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    using VectorF = Vector<float>;

    inline auto solveQuadraticEquation(double a, double b, double c) -> std::vector<double>
    {
        const double discriminat = b * b - 4 * a * c;
        if (discriminat < 0)
            return {};

        if (discriminat == 0)
            return {-b / 2 * a};

        const auto x1 = (-b - std::sqrt(discriminat)) / 2 * a;
        const auto x2 = (-b + std::sqrt(discriminat)) / 2 * a;
        return {x1, x2};
    }

    struct Camera
    {
        float fovy; // in degree
        VectorF position;
        VectorF view;
        VectorF up;
    };

    struct Sphere
    {
        VectorF center;
        float radius;
    };

    struct Triangle : std::array<VectorF, 3>
    {
    };

    struct PreparedTriangle
    {
        VectorF vertex0;
        VectorF edge1;
        VectorF edge2;

        auto normal() const -> VectorF
        {
            return cross(edge1, edge2).normalized();
        }
    };

    auto prepare(Triangle t) -> PreparedTriangle
    {
        return {t[0], t[1] - t[0], t[2] - t[0]};
    }

    struct Scene
    {
        Camera camera;
        std::vector<Sphere> spheres;
        TriangleView triangles;
    };

    class Image
    {
    public:
        using Pixel = Vector<unsigned char>;

        Image(unsigned int width, unsigned int height) : width(width), height(height), pixels(width * height)
        {
        }

        auto operator()(unsigned int x, unsigned int y) -> Pixel&
        {
            return pixels[y * width + x];
        }

        auto operator()(unsigned int x, unsigned int y) const -> const Pixel&
        {
            return pixels[y * width + x];
        }

        void write(const std::filesystem::path& filename) const
        {
            if (!stbi_write_png(filename.string().c_str(), width, height, 3, pixels.data(), 0))
                throw std::runtime_error("Failed to write image " + filename.string());
        }

    private:
        unsigned int width;
        unsigned int height;
        std::vector<Pixel> pixels;
    };

    struct Intersection
    {
        float distance;
        VectorF point;
        VectorF normal;
    };

    struct Ray
    {
        VectorF origin;
        VectorF direction;
    };

    auto createRay(const Camera& camera, unsigned int width, unsigned int height, unsigned int x, unsigned int y) -> Ray
    {
        // we imagine a plane with the image just 1 before the camera, and then we shoot at those pixels

        const auto center = camera.position + camera.view;
        const auto xVec = cross(camera.view, camera.up);
        const auto yVec = camera.up;

        const auto delta = (std::tan(camera.fovy * std::numbers::pi_v<float> / 180.0f) * 2) / (height - 1);
        const auto xDeltaVec = xVec * delta;
        const auto yDeltaVec = yVec * delta;

        const auto xRel = (x - static_cast<float>(width - 1) / 2);
        const auto yRel = (y - static_cast<float>(height - 1) / 2);

        const auto pixel = center + xDeltaVec * xRel + yDeltaVec * yRel;

        Ray r;
        r.origin = center;
        r.direction = (pixel - camera.position).normalized();

        assert(!std::isnan(r.direction[0]) && !std::isnan(r.direction[1]) && !std::isnan(r.direction[2]));
        // std::cout << r.direction << std::endl;

        return r;
    }

    auto intersect(const Ray& ray, const Sphere& sphere) -> std::optional<Intersection>
    {
        // from
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

        // solve quadratic equation
        const auto a = 1;
        const auto b = 2 * dot(ray.direction, (ray.origin - sphere.center));
        const auto c = (ray.origin - sphere.center).lengthSqr() - sphere.radius * sphere.radius;

        const auto solutions = solveQuadraticEquation(a, b, c);
        if (solutions.empty())
            return {};

        // report the closer intersection
        const auto t = static_cast<float>(*std::min_element(std::begin(solutions), std::end(solutions)));

        Intersection inter;
        inter.distance = t;
        inter.point = ray.origin + t * ray.direction;
        inter.normal = (inter.point - sphere.center).normalized();
        return inter;
    }

    // modified Möller and Trumbore's version
    auto intersect(const Ray& ray, const PreparedTriangle& triangle) -> std::optional<Intersection>
    {
        constexpr auto epsilon = 0.000001f;

        const auto pvec = cross(ray.direction, triangle.edge2);
        const auto det = dot(triangle.edge1, pvec);
        if (det > -epsilon && det < epsilon)
            return {};

        const auto inv_det = 1.0f / det;
        const auto tvec = ray.origin - triangle.vertex0;
        const auto u = dot(tvec, pvec) * inv_det;
        if (u < 0.0f || u > 1.0f)
            return {};

        const auto qvec = cross(tvec, triangle.edge1);
        const auto v = dot(ray.direction, qvec) * inv_det;
        if (v < 0.0f || u + v >= 1.0f)
            return {};
        const auto t = dot(triangle.edge2, qvec) * inv_det;
        if (t < 0)
            return {};

        return Intersection{t, ray.origin + ray.direction * t, triangle.normal()};
    }

    auto colorByIntersectionNormal(std::optional<Intersection> hit) -> Image::Pixel
    {
        if (hit)
        {
            Image::Pixel r;
            for (int i = 0; i < 3; i++)
                r[i] = static_cast<unsigned char>(std::abs(hit->normal[i]) * 255);
            return r;
        }
        else
            return {}; // black
    }

    auto blendAndColorByIntersectionNormal(std::vector<Intersection> hits) -> Image::Pixel
    {
        constexpr auto translucency = 0.5f;

        std::sort(std::begin(hits), std::end(hits), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        auto t = translucency;
        Image::Pixel r;
        for (const auto& hit : hits)
        {
            // each hit contributes to the color, lesser with each iteration
            for (int i = 0; i < 3; i++)
                r[i] += static_cast<unsigned char>(std::abs(t * hit.normal[i]) * 255);
            t *= translucency;
        }

        for (int i = 0; i < 3; i++)
            r[i] = std::clamp<unsigned char>(r[i], 0, 255);

        return r;
    }

    constexpr auto blendIntersections = false;

    auto raycast(const Scene& scene, unsigned int width, unsigned int height) -> Image
    {
        Image img(width, height);

        for (auto y = 0u; y < height; y++)
        {
            for (auto x = 0u; x < width; x++)
            {
                const auto ray = createRay(scene.camera, width, height, x, height - 1 - y); // flip

                if constexpr (blendIntersections)
                {
                    std::vector<Intersection> hits;
                    for (const auto& sphere : scene.spheres)
                        if (const auto hit = intersect(ray, sphere))
                            hits.push_back(*hit);
                    for (const auto i : llama::ArrayDomainIndexRange{scene.triangles.mapping.arrayDomainSize})
                        if (const auto hit = intersect(ray, scene.triangles[i].loadAs<PreparedTriangle>()))
                            hits.push_back(*hit);
                    img(x, y) = blendAndColorByIntersectionNormal(hits);
                }
                else
                {
                    std::optional<Intersection> nearestHit;
                    auto updateNearestHit = [&](auto hit) {
                        if (!nearestHit || hit->distance < nearestHit->distance)
                            nearestHit = hit;
                    };
                    for (const auto& sphere : scene.spheres)
                        if (const auto hit = intersect(ray, sphere))
                            updateNearestHit(hit);
                    for (const auto i : llama::ArrayDomainIndexRange{scene.triangles.mapping.arrayDomainSize})
                        if (const auto hit = intersect(ray, scene.triangles[i].loadAs<PreparedTriangle>()))
                            updateNearestHit(hit);
                    img(x, y) = colorByIntersectionNormal(nearestHit);
                }
            }
        }

        return img;
    }

    auto lookAt(float fovy, VectorF pos, VectorF lookAt, VectorF up) -> Camera
    {
        const auto view = (lookAt - pos).normalized();
        const auto up2 = cross(view, cross(view, up)).normalized();
        return Camera{fovy, pos, view, up};
    }

    // auto cubicBallsScene() -> Scene
    //{
    //    const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{};
    //    for (auto z = -2; z <= 2; z++)
    //        for (auto y = -2; y <= 2; y++)
    //            for (auto x = -2; x <= 2; x++)
    //                spheres.push_back(Sphere{{(float) x, (float) y, (float) z}, 0.8f});
    //    return Scene{camera, std::move(spheres)};
    //}

    // auto axisBallsScene() -> Scene
    //{
    //    const auto camera = lookAt(45, {5, 5, 10}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{
    //        {{0, 0, 0}, 3.0f},
    //        {{0, 0, 5}, 2.0f},
    //        {{0, 5, 0}, 2.0f},
    //        {{5, 0, 0}, 2.0f},
    //        {{0, 0, -5}, 1.0f},
    //        {{0, -5, 0}, 1.0f},
    //        {{-5, 0, 0}, 1.0f}};
    //    return Scene{camera, std::move(spheres)};
    //}

    // auto randomSphereScene() -> Scene
    //{
    //    constexpr auto count = 1024;

    //    const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
    //    auto spheres = std::vector<Sphere>{};

    //    std::default_random_engine eng;
    //    std::uniform_real_distribution d{-2.0f, 2.0f};
    //    for (auto i = 0; i < count; i++)
    //        spheres.push_back({{d(eng), d(eng), d(eng)}, 0.2f});
    //    return Scene{camera, std::move(spheres)};
    //}

    //// not the original one, but a poor attempt
    // auto cornellBox() -> Scene
    //{
    //    Scene scene;
    //    scene.camera = lookAt(45, {0, 0, 7}, {0, 0, 0}, {0, 1, 0});
    //    scene.spheres.push_back({{-2.5f, -2.5f, -2.5f}, 1.5f});
    //    scene.spheres.push_back({{2.5f, -2.5f, 0.0f}, 1.5f});
    //    // back plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, -5}, {5, -5, -5}, {5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, -5}, {5, 5, -5}, {-5, 5, -5}}}}));
    //    // left plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, -5, -5}, {-5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, 5, -5}, {-5, 5, 5}}}}));
    //    // right plane
    //    scene.triangles.push_back(prepare(Triangle{{{{5, -5, 5}, {5, 5, -5}, {5, -5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{5, -5, 5}, {5, 5, 5}, {5, 5, -5}}}}));
    //    // bottom plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {-5, -5, -5}, {5, -5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, -5, 5}, {5, -5, -5}, {5, -5, 5}}}}));
    //    // top plane
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, 5, 5}, {5, 5, -5}, {-5, 5, -5}}}}));
    //    scene.triangles.push_back(prepare(Triangle{{{{-5, 5, 5}, {5, 5, 5}, {5, 5, -5}}}}));

    //    return scene;
    //}

    auto sponzaScene(const char* objFile) -> Scene
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        const bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFile);
        std::cout << warn << std::endl;
        std::cerr << err << std::endl;
        if (!ret)
            throw std::runtime_error{"Failed to load sponza scene"};

        Scene scene;
        scene.camera = lookAt(45, {200, 100, 0}, {0, 100, 0}, {0, 1, 0});
        scene.spheres.push_back({{-30.0f, 30.0f, -30.0f}, 30.0f});
        scene.spheres.push_back({{30.0f, 30.0f, 30.0f}, 30.0f});

        const auto triangleCount
            = std::accumulate(begin(shapes), end(shapes), size_t{0}, [](size_t acc, const auto& shape) {
                  return acc
                      + std::count(begin(shape.mesh.num_face_vertices), std::end(shape.mesh.num_face_vertices), 3);
              });
        auto mapping = Mapping{llama::ArrayDomain{triangleCount}};
        scene.triangles = llama::allocView(mapping);

        size_t write = 0;
        for (const auto& shape : shapes)
        {
            const auto& mesh = shape.mesh;

            size_t indexOffset = 0;
            for (const auto vertexCount : mesh.num_face_vertices)
            {
                if (vertexCount == 3)
                {
                    Triangle t;
                    for (const auto v : {0, 1, 2})
                    {
                        const tinyobj::index_t idx = mesh.indices[indexOffset + v];
                        for (const auto c : {0, 1, 2})
                            t[v][c] = attrib.vertices[3 * idx.vertex_index + c];
                    }
                    const auto pt = prepare(t);
                    scene.triangles[write](Vertex0{}, X{}) = pt.vertex0[0];
                    scene.triangles[write](Vertex0{}, Y{}) = pt.vertex0[1];
                    scene.triangles[write](Vertex0{}, Z{}) = pt.vertex0[2];
                    scene.triangles[write](Edge1{}, X{}) = pt.edge1[0];
                    scene.triangles[write](Edge1{}, Y{}) = pt.edge1[1];
                    scene.triangles[write](Edge1{}, Z{}) = pt.edge1[2];
                    scene.triangles[write](Edge2{}, X{}) = pt.edge2[0];
                    scene.triangles[write](Edge2{}, Y{}) = pt.edge2[1];
                    scene.triangles[write](Edge2{}, Z{}) = pt.edge2[2];
                    write++;
                }
                indexOffset += vertexCount;
            }
        }

        return scene;
    }
} // namespace

int main(int argc, const char* argv[])
try
{
    const auto width = 160;
    const auto height = 120;

    // const auto scene = loadScene(sceneFile);
    // const auto scene = cubicBallsScene();
    // const auto scene = axisBallsScene();
    // const auto scene = randomSphereScene();
    // const auto scene = cornellBox();
    if (argc != 2)
    {
        std::cerr << "Please pass the location of sponza.obj as argument. The Sponza scene is available as git "
                     "submodule inside <gitrepo>/examples/raycast/Sponza\n";
    }
    const auto scene = sponzaScene(argv[1]);

    const auto start = std::chrono::high_resolution_clock::now();
    const auto image = raycast(scene, width, height);
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Raycast took " << std::chrono::duration<double>(end - start).count() << "s\n";

    image.write("out.png");
    std::system("./out.png");
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
    return 2;
}
