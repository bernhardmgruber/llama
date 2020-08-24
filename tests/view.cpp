#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Value {};
}

using DatumDomain = llama::DS<
    llama::DE<tag::Value, int>
>;
// clang-format on

TEST_CASE("view default ctor")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    llama::View<llama::mapping::SoA<UserDomain, DatumDomain>, std::byte *>
        view1;
    llama::View<llama::mapping::AoS<UserDomain, DatumDomain>, std::byte *>
        view2;
    llama::View<llama::mapping::One<UserDomain, DatumDomain>, std::byte *>
        view3;
    llama::View<
        llama::mapping::tree::Mapping<UserDomain, DatumDomain, llama::Tuple<>>,
        std::byte *>
        view4;
}

TEST_CASE("view move")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view1 = allocView(Mapping(viewSize));

    decltype(view1) view2;
    view1({3, 3}) = 1;
    view2 = std::move(view1);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view swap")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view1 = allocView(Mapping(viewSize));
    auto view2 = allocView(Mapping(viewSize));

    view1({3, 3}) = 1;
    view2({3, 3}) = 2;

    std::swap(view1, view2);

    CHECK(view1({3, 3}) == 2);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view allocator vector")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::Vector{});

    for(auto i : llama::UserDomainCoordRange{viewSize}) view(i) = 42;
}

TEST_CASE("view allocator shared_ptr")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::SharedPtr{});

    for(auto i : llama::UserDomainCoordRange{viewSize}) view(i) = 42;
}

TEST_CASE("view allocator Stack")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(
        Mapping(viewSize),
        llama::allocator::Stack<16 * 16 * llama::SizeOf<DatumDomain>::value>{});

    for(auto i : llama::UserDomainCoordRange{viewSize}) view(i) = 42;
}