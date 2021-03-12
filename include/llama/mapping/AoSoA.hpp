// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
    /// \tparam BlockAccessOnly If true, the mapping will return the same reference to the first lane of a block for
    /// all elements mapped into the same block. This simplifies the mapping function in case the AoSoA mapping is used
    /// in conjunction with a vectorization library.
    /// \tparam LinearizeArrayDomainFunctor Defines how the user domain should be mapped into linear numbers and how big
    /// the linear domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        std::size_t Lanes,
        bool BlockAccessOnly = false,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct AoSoA
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        constexpr AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr AoSoA(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const auto flatArrayIndex = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize);
            const auto blockIndex = flatArrayIndex / Lanes;
            auto offset = (sizeOf<DatumDomain> * Lanes) * blockIndex
                + offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>> * Lanes;
            if constexpr (!BlockAccessOnly)
            {
                const auto laneIndex = flatArrayIndex % Lanes;
                offset += sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>) * laneIndex;
            }
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };

    template <
        std::size_t Lanes,
        bool BlockAccessOnly = false,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredAoSoA
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type = AoSoA<ArrayDomain, DatumDomain, Lanes, BlockAccessOnly, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping