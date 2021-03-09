// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
    inline constexpr auto dynamic = std::numeric_limits<std::size_t>::max();

    /// Represents a coordinate for an element inside the datum domain tree.
    /// \tparam Coords... the compile time coordinate.
    template <std::size_t... Coords>
    struct DatumCoord
    {
        /// The list of integral coordinates as `boost::mp11::mp_list`.
        using List = boost::mp11::mp_list_c<std::size_t, Coords...>;

        static constexpr std::size_t front = boost::mp11::mp_front<List>::value;
        static constexpr std::size_t back = boost::mp11::mp_back<List>::value;
        static constexpr std::size_t size = sizeof...(Coords);
    };

    template <>
    struct DatumCoord<>
    {
        using List = boost::mp11::mp_list_c<std::size_t>;

        static constexpr std::size_t size = 0;
    };

    inline namespace literals
    {
        template <char... Digits>
        constexpr auto operator"" _DC()
        {
            constexpr auto coord = []() constexpr
            {
                char digits[] = {(Digits - 48)...};
                std::size_t acc = 0;
                std ::size_t powerOf10 = 1;
                for (int i = sizeof...(Digits) - 1; i >= 0; i--)
                {
                    acc += digits[i] * powerOf10;
                    powerOf10 *= 10;
                }
                return acc;
            }
            ();
            return DatumCoord<coord>{};
        }
    } // namespace literals

    namespace internal
    {
        template <class L>
        struct mp_unwrap_sizes_impl;

        template <template <class...> class L, typename... T>
        struct mp_unwrap_sizes_impl<L<T...>>
        {
            using type = DatumCoord<T::value...>;
        };

        template <typename L>
        using mp_unwrap_sizes = typename mp_unwrap_sizes_impl<L>::type;
    } // namespace internal

    /// Converts a type list of integral constants into a \ref DatumCoord.
    template <typename L>
    using DatumCoordFromList = internal::mp_unwrap_sizes<L>;

    /// Concatenate two \ref DatumCoord.
    template <typename DatumCoord1, typename DatumCoord2>
    using Cat = DatumCoordFromList<boost::mp11::mp_append<typename DatumCoord1::List, typename DatumCoord2::List>>;

    /// Concatenate two \ref DatumCoord instances.
    template <typename DatumCoord1, typename DatumCoord2>
    auto cat(DatumCoord1, DatumCoord2)
    {
        return Cat<DatumCoord1, DatumCoord2>{};
    }

    /// DatumCoord without first coordinate component.
    template <typename DatumCoord>
    using PopFront = DatumCoordFromList<boost::mp11::mp_pop_front<typename DatumCoord::List>>;

    namespace internal
    {
        template <typename First, typename Second>
        struct DatumCoordCommonPrefixIsBiggerImpl;

        template <std::size_t... Coords1, std::size_t... Coords2>
        struct DatumCoordCommonPrefixIsBiggerImpl<DatumCoord<Coords1...>, DatumCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                {
                    if (a1[i] > a2[i])
                        return true;
                    if (a1[i] < a2[i])
                        return false;
                }
                return false;
            }
            ();
        };
    } // namespace internal

    /// Checks wether the first DatumCoord is bigger than the second.
    template <typename First, typename Second>
    inline constexpr auto DatumCoordCommonPrefixIsBigger
        = internal::DatumCoordCommonPrefixIsBiggerImpl<First, Second>::value;

    namespace internal
    {
        template <typename First, typename Second>
        struct DatumCoordCommonPrefixIsSameImpl;

        template <std::size_t... Coords1, std::size_t... Coords2>
        struct DatumCoordCommonPrefixIsSameImpl<DatumCoord<Coords1...>, DatumCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for (auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                    if (a1[i] != a2[i])
                        return false;
                return true;
            }
            ();
        };
    } // namespace internal

    /// Checks wether two \ref DatumCoord are the same or one is the prefix of
    /// the other.
    template <typename First, typename Second>
    inline constexpr auto DatumCoordCommonPrefixIsSame
        = internal::DatumCoordCommonPrefixIsSameImpl<First, Second>::value;
} // namespace llama
