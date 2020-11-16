/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Stopwatch.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING = 3; ///< 0 AoS, 1 SoA, 2 SoA (separate blobs), 4 AoSoA, 4 tree AoS, 5 tree SoA
constexpr auto KERNEL = 2; ///< 0 non-caching kernel, 1 kernel with shared memory caching, 2 non-caching Vc kernel
constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto ALLOW_RSQRT = true; // rsqrt can be way faster, but less accurate

using FP = float;
constexpr FP EPS2 = 0.01;

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
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Mass, FP>>;
// clang-format on

template <typename Acc, typename VirtualParticleI, typename VirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(const Acc& acc, VirtualParticleI pi, VirtualParticleJ pj, FP ts)
{
    auto dist = pi(tag::Pos()) - pj(tag::Pos());
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X()) + dist(tag::Y()) + dist(tag::Z());
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = ALLOW_RSQRT ? alpaka::math::rsqrt(acc, distSixth) : (1.0f / std::sqrt(distSixth));
    const FP sts = pj(tag::Mass()) * invDistCube * ts;
    pi(tag::Vel()) += dist * sts;
}

template <std::size_t ProblemSize, std::size_t Elems, std::size_t BlockSize>
struct UpdateKernelSM
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        auto sharedView = [&] {
            const auto sharedMapping = llama::mapping::SoA(
                typename View::ArrayDomain{BlockSize},
                typename View::DatumDomain{}); // bug: nvcc 11.1 cannot have {} to call ctor

            // if there is only 1 thread per block, avoid using shared
            // memory
            if constexpr (BlockSize / Elems == 1)
                return llama::allocViewStack<View::ArrayDomain::rank, typename View::DatumDomain>();
            else
            {
                constexpr auto sharedMemSize = llama::sizeOf<typename View::DatumDomain> * BlockSize;
                auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMemSize], __COUNTER__>(acc);
                return llama::View{
                    sharedMapping,
                    llama::Array<std::byte*, 1>{
                        &sharedMem[0]}}; // bug: nvcc 11.1 needs explicit template args for llama::Array
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);
        LLAMA_INDEPENDENT_DATA
        for (std::size_t b = 0; b < (ProblemSize + BlockSize - 1u) / BlockSize; ++b)
        {
            const auto start2 = b * BlockSize;
            const auto end2 = alpaka::math::min(acc, start2 + BlockSize, ProblemSize) - start2;

            LLAMA_INDEPENDENT_DATA
            for (auto pos2 = decltype(end2)(0); pos2 + ti < end2; pos2 += BlockSize / Elems)
                sharedView(pos2 + tbi) = particles(start2 + pos2 + tbi);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for (auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2)
            {
                LLAMA_INDEPENDENT_DATA
                for (auto i = start; i < end; ++i)
                    pPInteraction(acc, particles(i), sharedView(pos2), ts);
            }
            alpaka::syncBlockThreads(acc);
        }
    }
};

template <std::size_t ProblemSize, std::size_t Elems>
struct UpdateKernel
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for (auto j = 0; j < ProblemSize; ++j)
        {
            LLAMA_INDEPENDENT_DATA
            for (auto i = start; i < end; ++i)
                pPInteraction(acc, particles(i), particles(j), ts);
        }
    }
};

#if __has_include(<Vc/Vc>)
#    include <Vc/Vc>

template <std::size_t Elems>
struct VecType
{
    using type = Vc::SimdArray<float, Elems>;
};
template <>
struct VecType<1>
{
    using type = float;
};

namespace aaa
{
    LLAMA_FN_HOST_ACC_INLINE float rsqrt(float f)
    {
        return 1.0f / std::sqrt(f);
    }
} // namespace aaa

template <std::size_t Elems, typename VirtualParticleI, typename VirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteractionVc(VirtualParticleI pi, VirtualParticleJ pj, FP ts)
{
    // using vec = Vc::Vector<float, Vc::abi::fixed<Elems>>;
    using vec = typename VecType<Elems>::type;

    // FIXME: this makes assumptions that there are always float_v::size() many elements blocked in the LLAMA view
    auto loadVec = [&](const float& src) {
        if constexpr (std::is_same_v<vec, float>)
            return src;
        else
            return vec(&src);
    };
    auto broadcastVec = [&](const float& src) { return vec(src); };

    auto storeVec = [&](float& dst, vec v) {
        if constexpr (std::is_same_v<vec, float>)
            dst = v;
        else
            v.store(&dst);
    };

    using aaa::rsqrt;
    using std::sqrt;
    using Vc::rsqrt;
    using Vc::sqrt;

    const vec xdistance = loadVec(pi(tag::Pos{}, tag::X{})) - broadcastVec(pj(tag::Pos{}, tag::X{}));
    const vec ydistance = loadVec(pi(tag::Pos{}, tag::Y{})) - broadcastVec(pj(tag::Pos{}, tag::Y{}));
    const vec zdistance = loadVec(pi(tag::Pos{}, tag::Z{})) - broadcastVec(pj(tag::Pos{}, tag::Z{}));
    const vec xdistanceSqr = xdistance * xdistance;
    const vec ydistanceSqr = ydistance * ydistance;
    const vec zdistanceSqr = zdistance * zdistance;
    const vec distSqr = +EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
    const vec distSixth = distSqr * distSqr * distSqr;
    const vec invDistCube = ALLOW_RSQRT ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const vec sts = broadcastVec(pj(tag::Mass())) * invDistCube * ts;
    storeVec(pi(tag::Vel{}, tag::X{}), xdistanceSqr * sts + loadVec(pi(tag::Vel{}, tag::X{})));
    storeVec(pi(tag::Vel{}, tag::Y{}), ydistanceSqr * sts + loadVec(pi(tag::Vel{}, tag::Y{})));
    storeVec(pi(tag::Vel{}, tag::Z{}), zdistanceSqr * sts + loadVec(pi(tag::Vel{}, tag::Z{})));
}

template <typename Mapping, std::size_t Elems>
inline constexpr auto canUseVcWithMapping = false;

template <typename ArrayDomain, typename DatumDomain, typename Linearize, std::size_t Elems>
inline constexpr auto canUseVcWithMapping<llama::mapping::SoA<ArrayDomain, DatumDomain, Linearize>, Elems> = true;

template <typename ArrayDomain, typename DatumDomain, std::size_t Lanes, typename Linearize, std::size_t Elems>
inline constexpr auto
    canUseVcWithMapping<llama::mapping::AoSoA<ArrayDomain, DatumDomain, Lanes, Linearize>, Elems> = Lanes
        >= Elems&& Lanes % Elems
    == 0;

template <std::size_t ProblemSize, std::size_t Elems>
struct UpdateKernelVc
{
    static_assert(ProblemSize % Elems == 0);

    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        static_assert(
            canUseVcWithMapping<typename View::Mapping, Elems>,
            "UpdateKernelVc only works with compatible mappings like SoA or AoSoAs");

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        LLAMA_INDEPENDENT_DATA
        for (auto j = std::size_t{0}; j < ProblemSize; ++j)
            pPInteractionVc<Elems>(particles(ti * Elems), particles(j), ts);
    }
};
#endif

template <std::size_t ProblemSize, std::size_t Elems>
struct MoveKernel
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for (auto i = start; i < end; ++i)
            particles(i)(tag::Pos()) += particles(i)(tag::Vel()) * ts;
    }
};

template <typename Acc>
struct Workdiv
{
#if __has_include(<Vc/Vc>)
    static constexpr std::size_t elements = Vc::float_v::size();
#else
    static constexpr std::size_t elements = 8;
#endif
    static constexpr std::size_t threadsPerBlock = 1;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <typename Dim, typename Size>
struct Workdiv<alpaka::AccGpuCudaRt<Dim, Size>>
{
    static constexpr std::size_t elements = 1;
    static constexpr std::size_t threadsPerBlock = 256;
};
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
template <typename Dim, typename Size>
struct Workdiv<alpaka::AccCpuOmp2Threads<Dim, Size>>
{
    // TODO: evaluate these previous settings from A. Matthes
    static constexpr std::size_t elements = 128;
    static constexpr std::size_t threadsPerBlock = 2;
};
#endif

using Dim = alpaka::DimInt<1>;
using Size = std::size_t;

using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
// using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
// using Acc = alpaka::AccCpuSerial<Dim, Size>;

using DevHost = alpaka::DevCpu;
using DevAcc = alpaka::Dev<Acc>;
using PltfHost = alpaka::Pltf<DevHost>;
using PltfAcc = alpaka::Pltf<DevAcc>;
using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

constexpr std::size_t elemCount = Workdiv<Acc>::elements;
constexpr std::size_t threadCount = Workdiv<Acc>::threadsPerBlock;
constexpr auto blockSize = elemCount * threadCount;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
constexpr auto aosoaLanes = 32; // coalesced memory access
#else
constexpr auto aosoaLanes = elemCount; // vectors
#endif

int main()
{
    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    constexpr FP ts = 0.0001;

    const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};

    const auto mapping = [&] {
        if constexpr (MAPPING == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (MAPPING == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (MAPPING == 2)
            return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
        if constexpr (MAPPING == 3)
            return llama::mapping::AoSoA<std::decay_t<decltype(arrayDomain)>, Particle, aosoaLanes>{arrayDomain};
        if constexpr (MAPPING == 4)
            return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Particle{}};
        if constexpr (MAPPING == 5)
            return llama::mapping::tree::Mapping{
                arrayDomain,
                llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                Particle{}};
    }();

    std::cout << PROBLEM_SIZE / 1000 << " thousand particles\n"
              << PROBLEM_SIZE * llama::sizeOf<Particle> / 1000 / 1000 << "MB \n";

    Stopwatch chrono;

    const auto bufferSize = Size(mapping.getBlobSize(0));

    auto hostBuffer = alpaka::allocBuf<std::byte, Size>(devHost, bufferSize);
    auto accBuffer = alpaka::allocBuf<std::byte, Size>(devAcc, bufferSize);

    chrono.printAndReset("alloc");

    auto hostView = llama::View{mapping, llama::Array{alpaka::getPtrNative(hostBuffer)}};
    auto accView = llama::View{mapping, llama::Array{alpaka::getPtrNative(accBuffer)}};

    chrono.printAndReset("views");

    /// Random initialization of the particles
    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        llama::One<Particle> temp;
        temp(tag::Pos(), tag::X()) = distribution(generator);
        temp(tag::Pos(), tag::Y()) = distribution(generator);
        temp(tag::Pos(), tag::Z()) = distribution(generator);
        temp(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        temp(tag::Mass()) = distribution(generator) / FP(100);
        hostView(i) = temp;
    }

    chrono.printAndReset("init");

    alpaka::memcpy(queue, accBuffer, hostBuffer, bufferSize);
    chrono.printAndReset("copy H->D");

    const alpaka::Vec<Dim, Size> Elems(static_cast<Size>(elemCount));
    const alpaka::Vec<Dim, Size> threads(static_cast<Size>(threadCount));
    const alpaka::Vec<Dim, Size> blocks(static_cast<Size>((PROBLEM_SIZE + blockSize - 1u) / blockSize));

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{blocks, threads, Elems};

    for (std::size_t s = 0; s < STEPS; ++s)
    {
        auto updateKernel = [&] {
            if constexpr (KERNEL == 0)
                return UpdateKernelSM<PROBLEM_SIZE, elemCount, blockSize>{};
            else if constexpr (KERNEL == 1)
                return UpdateKernel<PROBLEM_SIZE, elemCount>{};
#if __has_include(<Vc/Vc>)
            else if constexpr (KERNEL == 2)
                return UpdateKernelVc<PROBLEM_SIZE, elemCount>{};
#endif
        }();
        alpaka::exec<Acc>(queue, workdiv, updateKernel, accView, ts);
        chrono.printAndReset("update", '\t');

        MoveKernel<PROBLEM_SIZE, elemCount> moveKernel;
        alpaka::exec<Acc>(queue, workdiv, moveKernel, accView, ts);
        chrono.printAndReset("move");
    }

    alpaka::memcpy(queue, hostBuffer, accBuffer, bufferSize);
    chrono.printAndReset("copy D->H");

    return 0;
}
