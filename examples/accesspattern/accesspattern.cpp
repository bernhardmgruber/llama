#include <llama/llama.hpp>

constexpr auto PROBLEM_SIZE = 1024;

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
}

using Vector = llama::DS<
    llama::DE<tag::X, float>,
    llama::DE<tag::Y, float>,
    llama::DE<tag::Z, float>
>;
// clang-format on

template <typename T_ArrayDomain, typename T_DatumDomain>
struct AccessPatternDetection
{
    using ArrayDomain = T_ArrayDomain;
    using DatumDomain = T_DatumDomain;

    static constexpr std::size_t blobCount = 1;

    static constexpr auto tagCount = boost::mp11::mp_size<llama::FlattenDatumDomain<DatumDomain>>::value;
    std::array<bool, tagCount> accessedTags{};

    constexpr AccessPatternDetection() = default;

    constexpr AccessPatternDetection(ArrayDomain, DatumDomain = {})
    {
    }

    constexpr auto getBlobSize(std::size_t) const -> std::size_t
    {
        return 0;
    }

    template <std::size_t... DatumDomainCoord>
    constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> llama::NrAndOffset
    {
        constexpr auto i = llama::flatDatumCoord<DatumDomain, llama::DatumCoord<DatumDomainCoord...>>;
        accessedTags[i] = true;
        return {};
    }
};

template <typename View>
constexpr void add(const View& a, const View& b, View& c)
{
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
    {
        c(i)(tag::X{}) = a(i)(tag::X{}) + b(i)(tag::X{});
        c(i)(tag::Y{}) = a(i)(tag::Y{}) - b(i)(tag::Y{});
        c(i)(tag::Z{}) = a(i)(tag::Z{}) * b(i)(tag::Z{});
    }
}

constexpr auto detectAccessPattern()
{
    constexpr auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
    constexpr auto mapping = AccessPatternDetection{arrayDomain, Vector{}};

    constexpr auto view = allocView(mapping);

    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
        view[i](tag::Y{}) = i;

    return mapping.accessedTags;
}

int main(std::ofstream& plotFile)
{
    constexpr auto ap = detectAccessPattern();
    static_assert(ap[0] == false);
    static_assert(ap[1] == true);
    static_assert(ap[2] == false);
}
