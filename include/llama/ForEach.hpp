/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "DatumCoord.hpp"
#include "Functions.hpp"

#include <type_traits>

namespace llama
{
    namespace internal
    {
        template<
            typename T,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            T,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            using InnerDatumCoord = decltype(inner);
            if constexpr(
                InnerDatumCoord::size >= BaseDatumCoord::size
                && DatumCoordIsSame<InnerDatumCoord, BaseDatumCoord>)
                functor(
                    BaseDatumCoord{},
                    DatumCoordFromList<boost::mp11::mp_drop_c<
                        typename InnerDatumCoord::List,
                        BaseDatumCoord::size>>{});
        };

        template<
            typename... Children,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            DatumStruct<Children...>,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<
                boost::mp11::mp_iota_c<sizeof...(Children)>>([&](auto i) {
                constexpr auto childIndex = decltype(i)::value;
                using DatumElement = boost::mp11::
                    mp_at_c<DatumStruct<Children...>, childIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                applyFunctorForEachLeaf(
                    GetDatumElementType<DatumElement>{},
                    base,
                    llama::DatumCoord<InnerCoords..., childIndex>{},
                    std::forward<Functor>(functor));
            });
        }
    }

    /** Can be used to access a given functor for every leaf in a datum domain
     * given as \ref DatumStruct. Basically a helper function to iterate over a
     * datum domain at compile time without the need to recursively iterate
     * yourself. The given functor needs to implement the operator() with two
     * template parameters for the outer and the inner coordinate in the datum
     * domain tree. These coordinates are both a \ref DatumCoord , which can be
     * concatenated to one coordinate with \ref Cat and used to
     * access the data. \tparam DatumDomain the datum domain (\ref
     * DatumStruct) to iterate over \tparam DatumCoordOrFirstUID DatumCoord or
     * a UID to address the start node inside the datum domain tree. Will be
     * given to the functor as \ref DatumCoord as first template parameter.
     * \tparam RestUID... optional further UIDs for addressing the start node
     */

    template<typename DatumDomain, typename Functor, std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE void
    forEach(Functor && functor, DatumCoord<Coords...> coord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::applyFunctorForEachLeaf(
            DatumDomain{},
            coord,
            DatumCoord<>{},
            std::forward<Functor>(functor));
    }

    template<typename DatumDomain, typename Functor, typename... UIDs>
    LLAMA_FN_HOST_ACC_INLINE void forEach(Functor && functor, UIDs...)
    {
        forEach<DatumDomain>(
            std::forward<Functor>(functor),
            GetCoordFromUID<DatumDomain, UIDs...>{});
    }
}
