#include <chrono>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized

constexpr auto MAPPING
    = 1; ///< 0 native AoS, 1 native SoA, 2 tree AoS, 3 tree SoA
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

    template<typename VirtualDatum1, typename VirtualDatum2>
    LLAMA_FN_HOST_ACC_INLINE void
    pPInteraction(VirtualDatum1 && p1, VirtualDatum2 && p2, FP ts)
    {
        auto distance = p1(tag::Pos()) + p2(tag::Pos());
        distance *= distance;
        const FP distSqr = EPS2 + distance(tag::X()) + distance(tag::Y())
            + distance(tag::Z());
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2(tag::Mass()) * invDistCube;
        distance *= s * ts;
        p1(tag::Vel()) += distance;
    }

    template<typename T_View>
    void update(T_View & particles, FP ts)
    {
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles(j), particles(i), ts);
        }
    }

    template<typename T_View>
    void move(T_View & particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles(i)(tag::Pos()) += particles(i)(tag::Vel()) * ts;
    }

    int main(int argc, char ** argv)
    {
        constexpr FP ts = 0.0001f;

        using UserDomain = llama::UserDomain<1>;
        const UserDomain userDomain{PROBLEM_SIZE};

        const auto mapping = [&] {
            if constexpr(MAPPING == 0)
            {
                using Mapping = llama::mapping::AoS<UserDomain, Particle>;
                return Mapping(userDomain);
            }
            if constexpr(MAPPING == 1)
            {
                using Mapping = llama::mapping::SoA<UserDomain, Particle>;
                return Mapping(userDomain);
            }
            if constexpr(MAPPING == 2)
            {
                auto treeOperationList = llama::Tuple{};
                using Mapping = llama::mapping::tree::
                    Mapping<UserDomain, Particle, decltype(treeOperationList)>;
                return Mapping(userDomain, treeOperationList);
            }
            if constexpr(MAPPING == 3)
            {
                auto treeOperationList
                    = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
                using Mapping = llama::mapping::tree::
                    Mapping<UserDomain, Particle, decltype(treeOperationList)>;
                return Mapping(userDomain, treeOperationList);
            }
        }();

        auto particles = allocView(mapping);

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles LLAMA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took "
                  << std::chrono::duration<double>(stop - start).count()
                  << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> distribution(FP(0), FP(1));
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
            {
                particles(i)(tag::Pos(), tag::X()) = distribution(engine);
                particles(i)(tag::Pos(), tag::Y()) = distribution(engine);
                particles(i)(tag::Pos(), tag::Z()) = distribution(engine);
                particles(i)(tag::Vel(), tag::X())
                    = distribution(engine) / FP(10);
                particles(i)(tag::Vel(), tag::Y())
                    = distribution(engine) / FP(10);
                particles(i)(tag::Vel(), tag::Z())
                    = distribution(engine) / FP(10);
                particles(i)(tag::Mass()) = distribution(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took "
                      << std::chrono::duration<double>(stop - start).count()
                      << "s\n";
        }

        for(std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\n";
            }
        }

        return 0;
    }
}

namespace manualAoS
{
    struct Vec
    {
        FP x;
        FP y;
        FP z;

        auto operator*=(FP s) -> Vec &
        {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        auto operator*=(Vec v) -> Vec &
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        auto operator+=(Vec v) -> Vec &
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        friend auto operator+(Vec a, Vec b) -> Vec
        {
            return a += b;
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

    inline void pPInteraction(Particle & p1, const Particle & p2, FP ts)
    {
        auto distance = p1.pos + p2.pos;
        distance *= distance;
        const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2.mass * invDistCube;
        distance *= s * ts;
        p1.vel += distance;
    }

    void update(Particle * particles, FP ts)
    {
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles[j], particles[i], ts);
        }
    }

    void move(Particle * particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles[i].pos += particles[i].vel * ts;
    }

    int main(int argc, char ** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles AoS\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<Particle> particles(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took "
                  << std::chrono::duration<double>(stop - start).count()
                  << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> distribution(FP(0), FP(1));
            for(auto & p : particles)
            {
                p.pos.x = distribution(engine);
                p.pos.y = distribution(engine);
                p.pos.z = distribution(engine);
                p.vel.x = distribution(engine) / FP(10);
                p.vel.y = distribution(engine) / FP(10);
                p.vel.z = distribution(engine) / FP(10);
                p.mass = distribution(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took "
                      << std::chrono::duration<double>(stop - start).count()
                      << "s\n";
        }

        for(std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\n";
            }
        }

        return 0;
    }
}

namespace manualSoA
{
    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP & p1velx,
        FP & p1vely,
        FP & p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx + p2posx;
        auto ydistance = p1posy + p2posy;
        auto zdistance = p1posz + p2posz;
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

    void update(
        FP * posx,
        FP * posy,
        FP * posz,
        FP * velx,
        FP * vely,
        FP * velz,
        FP * mass,
        FP ts)
    {
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(
                    posx[j],
                    posy[j],
                    posz[j],
                    velx[j],
                    vely[j],
                    velz[j],
                    posx[i],
                    posy[i],
                    posz[i],
                    mass[i],
                    ts);
        }
    }

    void move(
        FP * posx,
        FP * posy,
        FP * posz,
        FP * velx,
        FP * vely,
        FP * velz,
        FP * mass,
        FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            posx[i] += velx[i] * ts;
            posy[i] += vely[i] * ts;
            posz[i] += velz[i] * ts;
        }
    }

    int main(int argc, char ** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << PROBLEM_SIZE / 1000 << " thousand particles SoA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<FP> posx(PROBLEM_SIZE);
        std::vector<FP> posy(PROBLEM_SIZE);
        std::vector<FP> posz(PROBLEM_SIZE);
        std::vector<FP> velx(PROBLEM_SIZE);
        std::vector<FP> vely(PROBLEM_SIZE);
        std::vector<FP> velz(PROBLEM_SIZE);
        std::vector<FP> mass(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took "
                  << std::chrono::duration<double>(stop - start).count()
                  << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> distribution(FP(0), FP(1));
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
            {
                posx[i] = distribution(engine);
                posy[i] = distribution(engine);
                posz[i] = distribution(engine);
                velx[i] = distribution(engine) / FP(10);
                vely[i] = distribution(engine) / FP(10);
                velz[i] = distribution(engine) / FP(10);
                mass[i] = distribution(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took "
                      << std::chrono::duration<double>(stop - start).count()
                      << "s\n";
        }

        for(std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(
                    posx.data(),
                    posy.data(),
                    posz.data(),
                    velx.data(),
                    vely.data(),
                    velz.data(),
                    mass.data(),
                    ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(
                    posx.data(),
                    posy.data(),
                    posz.data(),
                    velx.data(),
                    vely.data(),
                    velz.data(),
                    mass.data(),
                    ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   "
                          << std::chrono::duration<double>(stop - start).count()
                          << "s\n";
            }
        }

        return 0;
    }
}

int main(int argc, char ** argv)
{
    int r = 0;
    r += usellama::main(argc, argv);
    r += manualAoS::main(argc, argv);
    r += manualSoA::main(argc, argv);
    return r;
}
