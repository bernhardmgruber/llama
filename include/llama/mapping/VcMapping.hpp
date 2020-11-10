// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef __has_include(<Vc/Vc>)
#    include "../Functions.hpp"
#    include "../Types.hpp"
#    include "Common.hpp"

#    include <Vc/Vc>

namespace llama::mapping
{
    /// Array of struct of Vc vectors mapping. Used to create a \ref View via \ref
    /// allocView.
    /// \tparam LinearizeArrayDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct Vc
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        using LargestType = boost::mp11::mp_max_element<DatumDomainAsList<DatumDomain>, sizeOf>;
        constexpr auto Lanes = ::Vc::Vector<LargestType>::size();

        Vc() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Vc(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            using Type = GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>;
            const auto flatArrayIndex = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize);
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = 0; // flatArrayIndex % Lanes;
            const auto offset = (sizeOf<DatumDomain> * Lanes) * blockIndex
                + offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>> * Lanes + sizeof(Type) * laneIndex;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace llama::mapping

#endif