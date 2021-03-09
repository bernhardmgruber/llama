#include "../common/Stopwatch.hpp"

#include <boost/asio/ip/host_name.hpp>
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

constexpr auto PROBLEM_SIZE = 16 * 1024;
constexpr auto STEPS = 10;
constexpr auto PRINT_BLOCK_PLACEMENT = false;

using FP = float;
constexpr FP TIMESTEP = 0.0001f;
constexpr FP EPS2 = 0.01f;

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
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticle p1, VirtualParticle p2)
{
    auto dist = p1(tag::Pos{}) - p2(tag::Pos{});
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP s = p2(tag::Mass{}) * invDistCube;
    dist *= s * TIMESTEP;
    p1(tag::Vel{}) += dist;
}

template <typename View>
void update(View& particles)
{
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
            pPInteraction(particles(j), particles(i));
    }
}

template <typename View>
void move(View& particles)
{
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * TIMESTEP;
}

template <std::size_t Mapping, std::size_t Alignment>
void run(std::ostream& plotFile)
{
    std::cout << (Mapping == 0 ? "AoS" : Mapping == 1 ? "SoA" : "SoA MB") << ' ' << Alignment << "\n";

    constexpr FP ts = 0.0001f;

    auto mapping = [&] {
        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        if constexpr (Mapping == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (Mapping == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (Mapping == 2)
            return llama::mapping::SoA<decltype(arrayDomain), Particle, true>{arrayDomain};
    }();

    auto particles = llama::allocView(std::move(mapping), llama::allocator::Vector<Alignment>{});

    if constexpr (PRINT_BLOCK_PLACEMENT)
    {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> blobRanges;
        for (const auto& blob : particles.storageBlobs)
        {
            const auto blobSize = mapping.getBlobSize(blobRanges.size());
            std::cout << "\tBlob #" << blobRanges.size() << " from " << &blob[0] << " to " << &blob[0] + blobSize
                      << '\n';
            const auto start = reinterpret_cast<std::uint64_t>(&blob[0]);
            blobRanges.emplace_back(start, start + blobSize);
        }
        std::sort(begin(blobRanges), end(blobRanges), [](auto a, auto b) { return a.first < b.first; });
        std::cout << "\tDistances: ";
        for (auto i = 0; i < blobRanges.size() - 1; i++)
            std::cout << blobRanges[i + 1].first - blobRanges[i].first << ' ';
        std::cout << '\n';
        std::cout << "\tGaps: ";
        for (auto i = 0; i < blobRanges.size() - 1; i++)
            std::cout << blobRanges[i + 1].first - blobRanges[i].second << ' ';
        std::cout << '\n';
    }

    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP(0), FP(1));
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

    double sumUpdate = 0;
    Stopwatch watch;
    for (std::size_t s = 0; s < STEPS; ++s)
    {
        update(particles);
        sumUpdate += watch.printAndReset("update", '\t');
        move(particles);
        watch.printAndReset("move");
    }

    if (Mapping == 0)
        plotFile << Alignment;
    plotFile << '\t' << sumUpdate / STEPS << (Mapping == 2 ? '\n' : '\t');
}

int main()
{
    using namespace boost::mp11;

    std::ofstream plotFile{"nbody.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << "\"alignment\"\t\"AoS\"\t\"SoA\"\t\"SoA MB\"\n";

    mp_for_each<mp_iota_c<28>>([&](auto ae) {
        mp_for_each<mp_list_c<std::size_t, 0, 1, 2>>([&](auto m) {
            constexpr auto mapping = decltype(m)::value;
            constexpr auto alignment = std::size_t{1} << decltype(ae)::value;
            run<mapping, alignment>(plotFile);
        });
    });

    std::cout << "Plot with: ./nbody.sh\n";
    std::ofstream{"nbody.sh"} << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody CPU {0}k particles on {1}"
set style data lines
set xtics rotate by 90 right
set key out top center maxrows 3
set yrange [0:*]
plot 'nbody.tsv' using 2:xtic(1) ti col, '' using 3:xtic(1) ti col, '' using 4:xtic(1) ti col
)",
        PROBLEM_SIZE / 1000,
        boost::asio::ip::host_name());
}
