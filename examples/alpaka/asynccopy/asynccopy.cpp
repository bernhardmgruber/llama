/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file asynccopy.cpp
 *  \brief Asynchronous bluring example for LLAMA using ALPAKA.
 */

#include <iostream>
#include <list>
#include <utility>

constexpr auto ASYNCCOPY_ASYNC
    = true; ///< defines whether the data shall be processed asynchronously
constexpr auto ASYNCCOPY_SHARED
    = true; ///< defines whether shared memory shall be used
constexpr auto ASYNCCOPY_SAVE
    = true; ///< defines whether the resultion image shall be saved
constexpr auto ASYNCCOPY_CHUNK_COUNT = 4;

constexpr auto ASYNCCOPY_DEFAULT_IMG_X
    = 4096; /// width of the default image if no png is loaded
constexpr auto ASYNCCOPY_DEFAULT_IMG_Y
    = 4096; /// height of the default image if no png is loaded
constexpr auto ASYNCCOPY_KERNEL_SIZE
    = 8; /// radius of the blur kernel, the diameter is this times two plus one
constexpr auto ASYNCCOPY_CHUNK_SIZE
    = 512; /// size of each chunk to be processed per alpaka kernel
constexpr auto ASYNCCOPY_TOTAL_ELEMS_PER_BLOCK
    = 16; /// number of elements per direction(!) every block should process

#ifdef __CUDACC__
#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../common/AlpakaAllocator.hpp"
#include "../../common/AlpakaMemCopy.hpp"
#include "../../common/AlpakaThreadElemsDistribution.hpp"
#include "../../common/Chrono.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include <alpaka/alpaka.hpp>
#include <llama/llama.hpp>
#include <random>

using Element = float;

// clang-format off
namespace px
{
    struct R{};
    struct G{};
    struct B{};
}

/// real datum domain of the image pixel used on the host for loading and saving
using Pixel = llama::DS<
    llama::DE<px::R, Element>,
    llama::DE<px::G, Element>,
    llama::DE<px::B, Element>>;

/// datum domain used in the kernel to modify the image
using PixelOnAcc = llama::DS<
    llama::DE<px::R, Element>, // you can remove one here if you want to checkout the difference of the result image ;)
    llama::DE<px::G, Element>,
    llama::DE<px::B, Element>>;
// clang-format on

/** Alpaka kernel functor used to blur a small image living in the device memory
 *  using the \ref PixelOnAcc datum domain
 */
template<
    std::size_t elems,
    std::size_t kernelSize,
    std::size_t totalElemsPerBlock>
struct BlurKernel
{
    template<typename T_Acc, typename T_View>
    LLAMA_FN_HOST_ACC_INLINE void
    operator()(T_Acc const & acc, T_View oldImage, T_View newImage) const
    {
        auto const threadIdxInGrid
            = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        [[maybe_unused]] auto shared = [&] {
            if constexpr(ASYNCCOPY_SHARED)
            {
                // Using SoA for the shared memory
                auto treeOperationList
                    = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
                using SharedMapping = llama::mapping::tree::Mapping<
                    typename T_View::Mapping::UserDomain,
                    typename T_View::Mapping::DatumDomain,
                    decltype(treeOperationList)>;
                auto constexpr sharedChunkSize
                    = totalElemsPerBlock + 2 * kernelSize;
                SharedMapping const sharedMapping(
                    {sharedChunkSize, sharedChunkSize}, treeOperationList);
                using SharedFactory = llama::Factory<
                    SharedMapping,
                    common::allocator::AlpakaShared<
                        T_Acc,
                        llama::SizeOf<PixelOnAcc>::value * sharedChunkSize
                            * sharedChunkSize,
                        __COUNTER__>>;
                return SharedFactory::allocView(sharedMapping, acc);
            }
            else
                return int{}; // dummy
        }();

        [[maybe_unused]] auto const blockIndex
            = alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if constexpr(ASYNCCOPY_SHARED)
        {
            constexpr auto threadsPerBlock = totalElemsPerBlock / elems;
            auto const threadIdxInBlock
                = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);

            std::size_t const b_start[2]
                = {blockIndex[0] * totalElemsPerBlock + threadIdxInBlock[0],
                   blockIndex[1] * totalElemsPerBlock + threadIdxInBlock[1]};
            std::size_t const b_end[2] = {
                alpaka::math::min(
                    acc,
                    (blockIndex[0] + 1) * totalElemsPerBlock + 2 * kernelSize,
                    oldImage.mapping.userDomainSize[0]),
                alpaka::math::min(
                    acc,
                    (blockIndex[1] + 1) * totalElemsPerBlock + 2 * kernelSize,
                    oldImage.mapping.userDomainSize[1]),
            };
            LLAMA_INDEPENDENT_DATA
            for(auto y = b_start[0]; y < b_end[0]; y += threadsPerBlock)
                LLAMA_INDEPENDENT_DATA
            for(auto x = b_start[1]; x < b_end[1]; x += threadsPerBlock)
                shared(
                    y - blockIndex[0] * totalElemsPerBlock,
                    x - blockIndex[1] * totalElemsPerBlock)
                    = oldImage(y, x);
        }

        std::size_t const start[2]
            = {threadIdxInGrid[0] * elems, threadIdxInGrid[1] * elems};
        std::size_t const end[2] = {
            alpaka::math::min(
                acc,
                (threadIdxInGrid[0] + 1) * elems,
                oldImage.mapping.userDomainSize[0] - 2 * kernelSize),
            alpaka::math::min(
                acc,
                (threadIdxInGrid[1] + 1) * elems,
                oldImage.mapping.userDomainSize[1] - 2 * kernelSize),
        };

        LLAMA_INDEPENDENT_DATA
        for(auto y = start[0]; y < end[0]; ++y) LLAMA_INDEPENDENT_DATA
        for(auto x = start[1]; x < end[1]; ++x)
        {
            auto sum = llama::stackVirtualDatumAlloc<PixelOnAcc>();
            sum = 0;

            using ItType = long int;
            const ItType i_b_start = ASYNCCOPY_SHARED
                ? ItType(y) - ItType(blockIndex[0] * totalElemsPerBlock)
                : y;
            const ItType i_a_start = ASYNCCOPY_SHARED
                ? ItType(x) - ItType(blockIndex[1] * totalElemsPerBlock)
                : x;
            const ItType i_b_end = ASYNCCOPY_SHARED
                ? ItType(y + 2 * kernelSize + 1)
                    - ItType(blockIndex[0] * totalElemsPerBlock)
                : y + 2 * kernelSize + 1;
            const ItType i_a_end = ASYNCCOPY_SHARED
                ? ItType(x + 2 * kernelSize + 1)
                    - ItType(blockIndex[1] * totalElemsPerBlock)
                : x + 2 * kernelSize + 1;
            LLAMA_INDEPENDENT_DATA
            for(auto b = i_b_start; b < i_b_end; ++b) LLAMA_INDEPENDENT_DATA
            for(auto a = i_a_start; a < i_a_end; ++a)
            {
                if constexpr(ASYNCCOPY_SHARED)
                    sum += shared(std::size_t(b), std::size_t(a));
                else
                    sum += oldImage(std::size_t(b), std::size_t(a));
            }
            sum /= Element((2 * kernelSize + 1) * (2 * kernelSize + 1));
            newImage(y + kernelSize, x + kernelSize) = sum;
        }
    }
};

int main(int argc, char ** argv)
{
    constexpr std::size_t totalElemsPerBlock = ASYNCCOPY_TOTAL_ELEMS_PER_BLOCK;
    constexpr std::size_t chunkSize = ASYNCCOPY_CHUNK_SIZE;
    constexpr std::size_t chunkCount = ASYNCCOPY_CHUNK_COUNT;
    constexpr std::size_t kernelSize = ASYNCCOPY_KERNEL_SIZE;
    constexpr std::size_t hardwareThreads = 2; // relevant for OpenMP2Threads

    // ALPAKA
    using Dim = alpaka::dim::DimInt<2>;
    using Host = alpaka::acc::AccCpuSerial<Dim, size_t>;

    // using Acc = alpaka::acc::AccGpuCudaRt<Dim, size_t>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, size_t>;
    // using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, size_t>;
    // using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, size_t>;
    // using Acc = alpaka::acc::AccCpuOmp4<Dim, size_t>;

    using Queue = alpaka::queue::Queue<
        Acc,
        std::conditional_t<
            ASYNCCOPY_ASYNC,
            alpaka::queue::NonBlocking,
            alpaka::queue::Blocking>>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    std::vector<Queue> queue;
    for(std::size_t i = 0; i < chunkCount; ++i) queue.push_back(Queue(devAcc));

    // ASYNCCOPY
    std::size_t img_x = ASYNCCOPY_DEFAULT_IMG_X;
    std::size_t img_y = ASYNCCOPY_DEFAULT_IMG_Y;
    std::size_t buffer_x = ASYNCCOPY_DEFAULT_IMG_X + 2 * kernelSize;
    std::size_t buffer_y = ASYNCCOPY_DEFAULT_IMG_Y + 2 * kernelSize;
    using Distribution = common::
        ThreadsElemsDistribution<Acc, totalElemsPerBlock, hardwareThreads>;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;

    std::vector<unsigned char> image;
    std::string out_filename = "output.png";

    if(argc > 1)
    {
        int x = 0;
        int y = 0;
        int n = 3;
        unsigned char * data = stbi_load(argv[1], &x, &y, &n, 0);
        image.resize(x * y * 3);
        std::copy(data, data + image.size(), begin(image));
        stbi_image_free(data);
        img_x = x;
        img_y = y;
        buffer_x = x + 2 * kernelSize;
        buffer_y = y + 2 * kernelSize;

        if(argc > 2)
            out_filename = std::string(argv[2]);
    }

    // LLAMA
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomainSize{buffer_y, buffer_x};
    const UserDomain chunkUserDomain{
        chunkSize + 2 * kernelSize, chunkSize + 2 * kernelSize};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
    using HostMapping = llama::mapping::tree::
        Mapping<UserDomain, Pixel, decltype(treeOperationList)>;
    using DevMapping = llama::mapping::tree::
        Mapping<UserDomain, PixelOnAcc, decltype(treeOperationList)>;
    const HostMapping hostMapping(userDomainSize, treeOperationList);
    const DevMapping devMapping(chunkUserDomain, treeOperationList);

    using HostFactory = llama::
        Factory<HostMapping, common::allocator::Alpaka<DevHost, size_t>>;
    using HostChunkFactory = llama::
        Factory<DevMapping, common::allocator::Alpaka<DevHost, size_t>>;
    using DevFactory
        = llama::Factory<DevMapping, common::allocator::Alpaka<DevAcc, size_t>>;
    using MirrorFactory = llama::Factory<
        DevMapping,
        common::allocator::AlpakaMirror<DevAcc, size_t, DevMapping>>;

    std::cout << "Image size: " << img_x << ":" << img_y << '\n'
              << hostMapping.getBlobSize(0) * 2 / 1024 / 1024
              << " MB on device\n";

    Chrono chrono;

    auto host = HostFactory::allocView(hostMapping, devHost);
    std::vector<decltype(HostChunkFactory::allocView(devMapping, devHost))>
        hostChunk;
    std::vector<decltype(DevFactory::allocView(devMapping, devAcc))> devOld;
    decltype(devOld) devNew;
    std::vector<decltype(MirrorFactory::allocView(devMapping, devOld[0]))>
        mirrorOld;
    decltype(mirrorOld) mirrorNew;
    for(std::size_t i = 0; i < chunkCount; ++i)
    {
        hostChunk.push_back(HostChunkFactory::allocView(devMapping, devHost));
        devOld.push_back(DevFactory::allocView(devMapping, devAcc));
        devNew.push_back(DevFactory::allocView(devMapping, devAcc));
        mirrorOld.push_back(MirrorFactory::allocView(devMapping, devOld[i]));
        mirrorNew.push_back(MirrorFactory::allocView(devMapping, devNew[i]));
    }

    chrono.printAndReset("Alloc");

    if(image.empty())
    {
        image.resize(img_x * img_y * 3);
        std::default_random_engine generator;
        std::normal_distribution<Element> distribution{
            Element(0), // mean
            Element(0.5) // stddev
        };
        LLAMA_INDEPENDENT_DATA
        for(std::size_t y = 0; y < buffer_y; ++y)
            for(std::size_t x = 0; x < buffer_x; ++x)
            {
                host(y, x)(px::R()) = std::abs(distribution(generator));
                host(y, x)(px::G()) = std::abs(distribution(generator));
                host(y, x)(px::B()) = std::abs(distribution(generator));
            }
    }
    else
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t y = 0; y < buffer_y; ++y)
            for(std::size_t x = 0; x < buffer_x; ++x)
            {
                auto X = x;
                auto Y = y;
                if(X < kernelSize)
                    X = kernelSize;
                if(Y < kernelSize)
                    Y = kernelSize;
                if(X > img_x + kernelSize - 1)
                    X = img_x + kernelSize - 1;
                if(Y > img_y + kernelSize - 1)
                    Y = img_y + kernelSize - 1;
                const auto * pixel
                    = &image[((Y - kernelSize) * img_x + X - kernelSize) * 3];
                host(y, x)(px::R()) = Element(pixel[0]) / 255.;
                host(y, x)(px::G()) = Element(pixel[1]) / 255.;
                host(y, x)(px::B()) = Element(pixel[2]) / 255.;
            }
    }

    chrono.printAndReset("Init");
    const alpaka::vec::Vec<Dim, size_t> elems(
        static_cast<size_t>(elemCount), static_cast<size_t>(elemCount));
    const alpaka::vec::Vec<Dim, size_t> threads(
        static_cast<size_t>(threadCount), static_cast<size_t>(threadCount));
    const alpaka::vec::Vec<Dim, size_t> blocks(
        static_cast<size_t>(
            (chunkSize + totalElemsPerBlock - 1) / totalElemsPerBlock),
        static_cast<size_t>(
            (chunkSize + totalElemsPerBlock - 1) / totalElemsPerBlock));
    const alpaka::vec::Vec<Dim, size_t> chunks(
        static_cast<size_t>((img_y + chunkSize - 1) / chunkSize),
        static_cast<size_t>((img_x + chunkSize - 1) / chunkSize));

    auto const workdiv
        = alpaka::workdiv::WorkDivMembers<Dim, size_t>{blocks, threads, elems};

    using HostViewType = llama::VirtualView<decltype(host)>;
    struct VirtualHostElement
    {
        HostViewType virtualHost;
        const UserDomain validMiniSize;
    };
    std::list<VirtualHostElement> virtualHostList;
    for(std::size_t chunk_y = 0; chunk_y < chunks[0]; ++chunk_y)
        for(std::size_t chunk_x = 0; chunk_x < chunks[1]; ++chunk_x)
        {
            // Create virtual view with size of mini view
            const UserDomain validMiniSize{
                ((chunk_y < chunks[0] - 1) ? chunkSize
                                           : (img_y - 1) % chunkSize + 1)
                    + 2 * kernelSize,
                ((chunk_x < chunks[1] - 1) ? chunkSize
                                           : (img_x - 1) % chunkSize + 1)
                    + 2 * kernelSize};
            llama::VirtualView<decltype(host)> virtualHost(
                host,
                {chunk_y * chunkSize, chunk_x * chunkSize},
                validMiniSize);
            // Find free chunk stream
            std::size_t chunkNr = virtualHostList.size();
            if(virtualHostList.size() < chunkCount)
                virtualHostList.push_back({virtualHost, validMiniSize});
            else
            {
                bool not_found = true;
                while(not_found)
                {
                    auto chunkIt = virtualHostList.begin();
                    for(chunkNr = 0; chunkNr < chunkCount; ++chunkNr)
                    {
                        if(alpaka::queue::empty(queue[chunkNr]))
                        {
                            // Copy data back
                            LLAMA_INDEPENDENT_DATA
                            for(std::size_t y = 0; y
                                < (*chunkIt).validMiniSize[0] - 2 * kernelSize;
                                ++y)
                            {
                                LLAMA_INDEPENDENT_DATA
                                for(std::size_t x = 0;
                                    x < (*chunkIt).validMiniSize[1]
                                        - 2 * kernelSize;
                                    ++x)
                                    (*chunkIt).virtualHost(
                                        y + kernelSize, x + kernelSize)
                                        = hostChunk[chunkNr](
                                            y + kernelSize, x + kernelSize);
                            }
                            chunkIt = virtualHostList.erase(chunkIt);
                            virtualHostList.insert(
                                chunkIt, {virtualHost, validMiniSize});
                            not_found = false;
                            break;
                        }
                        chunkIt++;
                    }
                    if(not_found)
                        std::this_thread::sleep_for(
                            std::chrono::microseconds{1});
                }
            }
            // Copy data from virtual view to mini view
            LLAMA_INDEPENDENT_DATA
            for(std::size_t y = 0; y < validMiniSize[0]; ++y)
            {
                LLAMA_INDEPENDENT_DATA
                for(std::size_t x = 0; x < validMiniSize[1]; ++x)
                    hostChunk[chunkNr](y, x) = virtualHost(y, x);
            }
            alpakaMemCopy(
                devOld[chunkNr],
                hostChunk[chunkNr],
                chunkUserDomain,
                queue[chunkNr]);
            BlurKernel<elemCount, kernelSize, totalElemsPerBlock> blurKernel;
            alpaka::kernel::exec<Acc>(
                queue[chunkNr],
                workdiv,
                blurKernel,
                mirrorOld[chunkNr],
                mirrorNew[chunkNr]);
            alpakaMemCopy(
                hostChunk[chunkNr],
                devNew[chunkNr],
                userDomainSize,
                queue[chunkNr]);
        }

    // Wait for not finished tasks on accelerator
    auto chunkIt = virtualHostList.begin();
    for(std::size_t chunkNr = 0; chunkNr < chunkCount; ++chunkNr)
    {
        alpaka::wait::wait(queue[chunkNr]);
        // Copy data back
        LLAMA_INDEPENDENT_DATA
        for(std::size_t y = 0; y < (*chunkIt).validMiniSize[0] - 2 * kernelSize;
            ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t x = 0;
                x < (*chunkIt).validMiniSize[1] - 2 * kernelSize;
                ++x)
                (*chunkIt).virtualHost(y + kernelSize, x + kernelSize)
                    = hostChunk[chunkNr](y + kernelSize, x + kernelSize);
        }
        chunkIt++;
    }
    chrono.printAndReset("Blur kernel");

    if(ASYNCCOPY_SAVE)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t y = 0; y < img_y; ++y)
            for(std::size_t x = 0; x < img_x; ++x)
            {
                auto * pixel = &image[(y * img_x + x) * 3];
                pixel[0] = host(y + kernelSize, x + kernelSize)(px::R()) * 255.;
                pixel[1] = host(y + kernelSize, x + kernelSize)(px::G()) * 255.;
                pixel[2] = host(y + kernelSize, x + kernelSize)(px::B()) * 255.;
            }
        stbi_write_png(out_filename.c_str(), img_x, img_y, 3, image.data(), 0);
    }

    return 0;
}