#pragma once

#include "Common.hpp"

#include <array>
#include <vector>

namespace llama::mapping
{
    /// Forwards all calls to the inner mapping. Counts all accesses made to all bytes, allowing to extract a heatmap.
    /// \tparam Mapping The type of the inner mapping.
    template <typename Mapping, typename CountType = std::size_t>
    struct Heatmap
    {
        using ArrayDomain = typename Mapping::ArrayDomain;
        using DatumDomain = typename Mapping::DatumDomain;
        static constexpr std::size_t blobCount = Mapping::blobCount;

        constexpr Heatmap() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Heatmap(Mapping mapping) : mapping(mapping)
        {
            for (auto i = 0; i < blobCount; i++)
                byteHits[i].resize(getBlobSize(i));
        }

        Heatmap(const Heatmap&) = delete;
        auto operator=(const Heatmap&) -> Heatmap& = delete;

        Heatmap(Heatmap&&) noexcept = default;
        auto operator=(Heatmap&&) noexcept -> Heatmap& = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobSize(std::size_t i) const -> std::size_t
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.getBlobSize(i);
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE const auto nao
                = mapping.template getBlobNrAndOffset<DatumDomainCoord...>(coord);
            for (auto i = 0; i < sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>); i++)
                byteHits[nao.nr][nao.offset + i]++;
            return nao;
        }

        Mapping mapping;
        mutable std::array<std::vector<CountType>, blobCount> byteHits;
    };
} // namespace llama::mapping
