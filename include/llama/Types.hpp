// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"

#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
    /// Anonymous naming for a \ref DatumElement. Especially used for a \ref
    /// DatumArray.
    struct NoName
    {};

    /// The run-time specified user domain.
    /// \tparam Dim compile time dimensionality of the user domain
    template<std::size_t Dim>
    struct UserDomain : Array<std::size_t, Dim>
    {};

    static_assert(std::is_trivially_default_constructible_v<
                  UserDomain<1>>); // so UserDomain<1>{} will produce a zeroed
                                   // coord. Should hold for all dimensions, but
                                   // just checking for <1> here.

    template<typename... Args>
    UserDomain(Args...) -> UserDomain<sizeof...(Args)>;
}

namespace std
{
    template<size_t N>
    struct tuple_size<llama::UserDomain<N>> : integral_constant<size_t, N>
    {};

    template<size_t I, size_t N>
    struct tuple_element<I, llama::UserDomain<N>>
    {
        using type = size_t;
    };
}

namespace llama
{
    /// A list of \ref DatumElement which may be used to define a datum domain.
    template<typename... Leaves>
    using DatumStruct = boost::mp11::mp_list<Leaves...>;

    /// Shortcut alias for \ref DatumStruct.
    template<typename... Leaves>
    using DS = DatumStruct<Leaves...>;

    /// Datum domain tree node which may either be a leaf or refer to a child
    /// tree presented as another \ref DatumStruct or \ref DatumArray.
    /// \tparam Tag Name of the node. May be any type (struct, class).
    /// \tparam Type Type of the node. May be either another sub tree consisting
    /// of a nested \ref DatumStruct or \ref DatumArray or any other type making
    /// it a leaf of this type.
    template<typename Tag, typename Type>
    using DatumElement = boost::mp11::mp_list<Tag, Type>;

    /// Shortcut alias for \ref DatumElement.
    template<typename Identifier, typename Type>
    using DE = DatumElement<Identifier, Type>;

    /// Tag describing an index. Used to access members of a \ref DatumArray.
    template<std::size_t I>
    using Index = boost::mp11::mp_size_t<I>;

    /// An array of identical \ref DatumElement with \ref Index specialized on
    /// consecutive numbers. Can be used anywhere where \ref DatumStruct may
    /// used.
    /// \tparam Child Type to repeat. May be either a nested \ref
    /// DatumStruct or DatumArray or any other type making it an array of leaves
    /// of this type.
    /// \tparam Count Number of repetitions of ChildType.
    template<typename Child, std::size_t Count>
    struct DatumArray
    {};

    /// Shortcut alias for \ref DatumArray
    template<typename ChildType, std::size_t Count>
    using DA = DatumArray<ChildType, Count>;

    struct NrAndOffset
    {
        std::size_t nr;
        std::size_t offset;
    };
}
