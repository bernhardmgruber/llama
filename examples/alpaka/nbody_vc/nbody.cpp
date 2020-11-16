/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Stopwatch.hpp"

#include <Vc/Vc>
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING = 3; ///< 0 AoS, 1 SoA, 2 SoA (separate blobs), 4 AoSoA, 4 tree AoS, 5 tree SoA
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

namespace stdext
{
    LLAMA_FN_HOST_ACC_INLINE float rsqrt(float f)
    {
        return 1.0f / std::sqrt(f);
    }
} // namespace stdext

// FIXME: this makes assumptions that there are always float_v::size() many elements blocked in the LLAMA view
template <typename Vec>
inline auto load(const float& src)
{
    if constexpr (std::is_same_v<Vec, float>)
            return src;
        else
        return Vec(&src);
}

template <typename Vec>
inline auto broadcast(const float& src)
{
    return Vec(src);
}

template <typename Vec>
inline auto store(float& dst, Vec v)
{
    if constexpr (std::is_same_v<Vec, float>)
            dst = v;
        else
            v.store(&dst);
}

template <typename Vec, typename VirtualParticleI, typename VirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticleI pi, VirtualParticleJ pj, FP ts)
{
    using std::sqrt;
    using stdext::rsqrt;
    using Vc::rsqrt;
    using Vc::sqrt;

    const Vec xdistance = load<Vec>(pi(tag::Pos{}, tag::X{})) - broadcast<Vec>(pj(tag::Pos{}, tag::X{}));
    const Vec ydistance = load<Vec>(pi(tag::Pos{}, tag::Y{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Y{}));
    const Vec zdistance = load<Vec>(pi(tag::Pos{}, tag::Z{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Z{}));
    const Vec xdistanceSqr = xdistance * xdistance;
    const Vec ydistanceSqr = ydistance * ydistance;
    const Vec zdistanceSqr = zdistance * zdistance;
    const Vec distSqr = +EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
    const Vec distSixth = distSqr * distSqr * distSqr;
    const Vec invDistCube = ALLOW_RSQRT ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const Vec sts = broadcast<Vec>(pj(tag::Mass())) * invDistCube * ts;
    store<Vec>(pi(tag::Vel{}, tag::X{}), xdistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::X{})));
    store<Vec>(pi(tag::Vel{}, tag::Y{}), ydistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::Y{})));
    store<Vec>(pi(tag::Vel{}, tag::Z{}), zdistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::Z{})));
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

template <std::size_t ProblemSize, std::size_t Elems, std::size_t BlockSize>
struct UpdateKernel
{
    // makes our life easier for now
    static_assert(ProblemSize % Elems == 0);
    static_assert(ProblemSize % BlockSize == 0);

    using Vec = typename VecType<Elems>::type;

    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        static_assert(
            canUseVcWithMapping<typename View::Mapping, Elems>,
            "UpdateKernel only works with compatible mappings like SoA or AoSoAs");

        auto sharedView = [&] {
            const auto sharedMapping = llama::mapping::SoA(
                typename View::ArrayDomain{BlockSize},
                typename View::DatumDomain{}); // bug: nvcc 11.1 cannot have {} to call ctor

            // if there is only 1 thread per block, avoid using shared memory
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

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        LLAMA_INDEPENDENT_DATA
        for (std::size_t blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
        {
            LLAMA_INDEPENDENT_DATA
            for (auto j = tbi; j < BlockSize; j += BlockSize / Elems)
                sharedView(j) = particles(blockOffset + j);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for (auto j = std::size_t{0}; j < BlockSize; ++j)
                pPInteraction<Vec>(particles(ti * Elems), sharedView(j), ts);
            alpaka::syncBlockThreads(acc);
        }
    }
};

template <std::size_t ProblemSize, std::size_t Elems>
struct MoveKernel
{
    static_assert(ProblemSize % Elems == 0);

    using Vec = typename VecType<Elems>::type;

    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * Elems;

        store<Vec>(
            particles(i)(tag::Pos{}, tag::X{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::X{})) + load<Vec>(particles(i)(tag::Vel{}, tag::X{})) * ts);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Y{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Y{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Y{})) * ts);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Z{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Z{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Z{})) * ts);
    }
};

template <typename Acc>
struct Workdiv
{
    static constexpr std::size_t elements = Vc::float_v::size();
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
        auto updateKernel = UpdateKernel<PROBLEM_SIZE, elemCount, blockSize>{};
        alpaka::exec<Acc>(queue, workdiv, updateKernel, accView, ts);
        chrono.printAndReset("update", '\t');

        auto moveKernel = MoveKernel<PROBLEM_SIZE, elemCount>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, accView, ts);
        chrono.printAndReset("move");
    }

    alpaka::memcpy(queue, hostBuffer, accBuffer, bufferSize);
    chrono.printAndReset("copy D->H");

    return 0;
}
