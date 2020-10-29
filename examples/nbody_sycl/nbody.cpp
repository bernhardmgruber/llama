#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized

constexpr auto MAPPING = 0; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blos), 3 tree AoS, 4 tree SoA
constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 5; ///< number of steps to calculate

using FP = float;
constexpr FP EPS2 = 0.01f;

namespace usellama
{
    // clang-format off
    namespace tag
    {
        struct Pos{};
        struct Vel{};
        struct X{};
        struct Y{};
        struct Z{};
        struct Mass{};
    }

    using Particle = llama::DS<
        llama::DE<tag::Pos, llama::DS<
            llama::DE<tag::X, FP>,
            llama::DE<tag::Y, FP>,
            llama::DE<tag::Z, FP>
        >>,
        llama::DE<tag::Vel, llama::DS<
            llama::DE<tag::X, FP>,
            llama::DE<tag::Y, FP>,
            llama::DE<tag::Z, FP>
        >>,
        llama::DE<tag::Mass, FP>
    >;
    // clang-format on

    template <typename VirtualParticle>
    void pPInteraction(VirtualParticle p1, VirtualParticle p2, FP ts)
    {
        auto dist = p1(tag::Pos{}) - p2(tag::Pos{});
        dist *= dist;
        const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2(tag::Mass{}) * invDistCube;
        dist *= s * ts;
        p1(tag::Vel{}) += dist;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        auto mapping = [&] {
            if constexpr (MAPPING == 0)
                return llama::mapping::AoS{arrayDomain, Particle{}};
            if constexpr (MAPPING == 1)
                return llama::mapping::SoA{arrayDomain, Particle{}};
            if constexpr (MAPPING == 2)
                return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
            if constexpr (MAPPING == 3)
                return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Particle{}};
            if constexpr (MAPPING == 4)
                return llama::mapping::tree::Mapping{
                    arrayDomain,
                    llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                    Particle{}};
        }();

        // auto particles = llama::allocView(std::move(mapping));

        std::cout << "LLAMA SYCL\n";

        const auto start = std::chrono::high_resolution_clock::now();
        auto buffer = sycl::buffer<std::byte, 1>{sycl::range<1>{mapping.getBlobSize(0)}};
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        auto queue = sycl::queue{sycl::cpu_selector{}};
        std::cout << "running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));

            {
                llama::View particles{mapping, llama::Array{buffer.get_access<sycl::access::mode::write>()}};
                for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
                {
                    auto p = particles(i);
                    p(tag::Pos{}, tag::X{}) = dist(engine);
                    p(tag::Pos{}, tag::Y{}) = dist(engine);
                    p(tag::Pos{}, tag::Z{}) = dist(engine);
                    p(tag::Vel{}, tag::X{}) = dist(engine) / FP(10);
                    p(tag::Vel{}, tag::Y{}) = dist(engine) / FP(10);
                    p(tag::Vel{}, tag::Z{}) = dist(engine) / FP(10);
                    p(tag::Mass{}) = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        const auto range = sycl::range<1>{PROBLEM_SIZE};
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    llama::View particles{mapping, llama::Array{buffer.get_access<sycl::access::mode::read_write>(h)}};
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
                            pPInteraction(particles(it[0]), particles(i), ts);
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    llama::View particles{mapping, llama::Array{buffer.get_access<sycl::access::mode::read_write>(h)}};
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        particles(it[0])(tag::Pos{}) += particles(it[0])(tag::Vel{}) * ts;
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace usellama

namespace manualAoS
{
    struct Vec
    {
        FP x;
        FP y;
        FP z;

        auto operator*=(FP s) -> Vec&
        {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        auto operator*=(Vec v) -> Vec&
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        auto operator+=(Vec v) -> Vec&
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        auto operator-=(Vec v) -> Vec&
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        friend auto operator-(Vec a, Vec b) -> Vec
        {
            return a -= b;
        }

        friend auto operator*(Vec a, FP s) -> Vec
        {
            return a *= s;
        }

        friend auto operator*(Vec a, Vec b) -> Vec
        {
            return a *= b;
        }
    };

    using Pos = Vec;
    using Vel = Vec;

    struct Particle
    {
        Pos pos;
        Vel vel;
        FP mass;
    };

    inline void pPInteraction(Particle& p1, const Particle& p2, FP ts)
    {
        auto distance = p1.pos - p2.pos;
        distance *= distance;
        const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2.mass * invDistCube;
        distance *= s * ts;
        p1.vel += distance;
    }

    void update(Particle* particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles[j], particles[i], ts);
        }
    }

    void move(Particle* particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles[i].pos += particles[i].vel * ts;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "AoS SYCL\n";

        const auto start = std::chrono::high_resolution_clock::now();
        const auto range = sycl::range<1>{PROBLEM_SIZE};
        auto buffer = sycl::buffer<Particle, 1>{range};
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        auto queue = sycl::queue{sycl::cpu_selector{}};
        std::cout << "running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            {
                auto particles = buffer.get_access<sycl::access::mode::write>();
                for (std::size_t i = 0; i < particles.get_count(); i++)
                {
                    auto& p = particles[i];
                    p.pos.x = dist(engine);
                    p.pos.y = dist(engine);
                    p.pos.z = dist(engine);
                    p.vel.x = dist(engine) / FP(10);
                    p.vel.y = dist(engine) / FP(10);
                    p.vel.z = dist(engine) / FP(10);
                    p.mass = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto particles = buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
                            pPInteraction(particles[it[0]], particles[i], ts);
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto particles = buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        particles[it[0]].pos += particles[it[0]].vel * ts;
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoS

namespace manualSoA
{
    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        xdistance *= s * ts;
        ydistance *= s * ts;
        zdistance *= s * ts;
        p1velx += xdistance;
        p1vely += ydistance;
        p1velz += zdistance;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "SoA SYCL\n";

        const auto start = std::chrono::high_resolution_clock::now();
        const auto range = sycl::range<1>{PROBLEM_SIZE};
        auto posxBuffer = sycl::buffer<FP, 1>{range};
        auto posyBuffer = sycl::buffer<FP, 1>{range};
        auto poszBuffer = sycl::buffer<FP, 1>{range};
        auto velxBuffer = sycl::buffer<FP, 1>{range};
        auto velyBuffer = sycl::buffer<FP, 1>{range};
        auto velzBuffer = sycl::buffer<FP, 1>{range};
        auto massBuffer = sycl::buffer<FP, 1>{range};
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        auto queue = sycl::queue{sycl::cpu_selector{}};
        std::cout << "running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            {
                auto posx = posxBuffer.get_access<sycl::access::mode::write>();
                auto posy = posyBuffer.get_access<sycl::access::mode::write>();
                auto posz = poszBuffer.get_access<sycl::access::mode::write>();
                auto velx = velxBuffer.get_access<sycl::access::mode::write>();
                auto vely = velyBuffer.get_access<sycl::access::mode::write>();
                auto velz = velzBuffer.get_access<sycl::access::mode::write>();
                auto mass = massBuffer.get_access<sycl::access::mode::write>();
                for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
                {
                    posx[i] = dist(engine);
                    posy[i] = dist(engine);
                    posz[i] = dist(engine);
                    velx[i] = dist(engine) / FP(10);
                    vely[i] = dist(engine) / FP(10);
                    velz[i] = dist(engine) / FP(10);
                    mass[i] = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto posx = posxBuffer.get_access<sycl::access::mode::write>(h);
                    auto posy = posyBuffer.get_access<sycl::access::mode::write>(h);
                    auto posz = poszBuffer.get_access<sycl::access::mode::write>(h);
                    auto velx = velxBuffer.get_access<sycl::access::mode::write>(h);
                    auto vely = velyBuffer.get_access<sycl::access::mode::write>(h);
                    auto velz = velzBuffer.get_access<sycl::access::mode::write>(h);
                    auto mass = massBuffer.get_access<sycl::access::mode::write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
                            pPInteraction(
                                posx[it[0]],
                                posy[it[0]],
                                posz[it[0]],
                                velx[it[0]],
                                vely[it[0]],
                                velz[it[0]],
                                posx[i],
                                posy[i],
                                posz[i],
                                mass[i],
                                ts);
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto posx = posxBuffer.get_access<sycl::access::mode::write>(h);
                    auto posy = posyBuffer.get_access<sycl::access::mode::write>(h);
                    auto posz = poszBuffer.get_access<sycl::access::mode::write>(h);
                    auto velx = velxBuffer.get_access<sycl::access::mode::write>(h);
                    auto vely = velyBuffer.get_access<sycl::access::mode::write>(h);
                    auto velz = velzBuffer.get_access<sycl::access::mode::write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        posx[it[0]] += velx[it[0]] * ts;
                        posy[it[0]] += vely[it[0]] * ts;
                        posz[it[0]] += velz[it[0]] * ts;
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualSoA

namespace manualAoSoA
{
    constexpr auto LANES = 16;
    constexpr auto L1D_SIZE = 32 * 1024;

    struct alignas(64) ParticleBlock
    {
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } pos;
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } vel;
        FP mass[LANES];
    };

    constexpr auto BLOCKS_PER_TILE = 64; // L1D_SIZE / sizeof(ParticleBlock);
    constexpr auto BLOCKS = PROBLEM_SIZE / LANES;

    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        xdistance *= s * ts;
        ydistance *= s * ts;
        zdistance *= s * ts;
        p1velx += xdistance;
        p1vely += ydistance;
        p1velz += zdistance;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "AoSoA SYCL";
        std::cout << "\n";

        const auto start = std::chrono::high_resolution_clock::now();
        auto range = sycl::range<1>{BLOCKS};
        auto buffer = sycl::buffer<ParticleBlock, 1>{range};
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        auto queue = sycl::queue{sycl::cpu_selector{}};
        std::cout << "running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            {
                auto particles = buffer.get_access<sycl::access::mode::write>();
                for (std::size_t bi = 0; bi < BLOCKS; ++bi)
                {
                    auto& block = particles[bi];
                    for (std::size_t i = 0; i < LANES; ++i)
                    {
                        block.pos.x[i] = dist(engine);
                        block.pos.y[i] = dist(engine);
                        block.pos.z[i] = dist(engine);
                        block.vel.x[i] = dist(engine) / FP(10);
                        block.vel.y[i] = dist(engine) / FP(10);
                        block.vel.z[i] = dist(engine) / FP(10);
                        block.mass[i] = dist(engine) / FP(100);
                    }
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto particles = buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        const auto bi = it[0];
                        for (std::size_t i = 0; i < LANES; i++)
                            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                                for (std::size_t j = 0; j < LANES; j++)
                                {
                                    const auto& blockI = particles[bi];
                                    auto& blockJ = particles[bj];
                                    pPInteraction(
                                        blockJ.pos.x[j],
                                        blockJ.pos.y[j],
                                        blockJ.pos.z[j],
                                        blockJ.vel.x[j],
                                        blockJ.vel.y[j],
                                        blockJ.vel.z[j],
                                        blockI.pos.x[i],
                                        blockI.pos.y[i],
                                        blockI.pos.z[i],
                                        blockI.mass[i],
                                        ts);
                                }
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                queue.submit([&](sycl::handler& h) {
                    auto particles = buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(range, [=](sycl::item<1> it) mutable {
                        const auto bi = it[0];
                        auto& block = particles[bi];
                        for (std::size_t i = 0; i < LANES; ++i)
                        {
                            block.pos.x[i] += block.vel.x[i] * ts;
                            block.pos.y[i] += block.vel.y[i] * ts;
                            block.pos.z[i] += block.vel.z[i] * ts;
                        }
                    });
                });
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoSoA

int main(int argc, char** argv)
try
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles "
              << "(" << PROBLEM_SIZE * sizeof(float) * 7 / 1024 << "kiB)\n";

    int r = 0;
    r += usellama::main(argc, argv);
    r += manualAoS::main(argc, argv);
    r += manualSoA::main(argc, argv);
    r += manualAoSoA::main(argc, argv);
    return r;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << "\n";
}
