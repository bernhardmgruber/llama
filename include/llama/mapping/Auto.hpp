#pragma once

#include "../Types.hpp"
#include "AoS.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename UserDomain, typename DatumDomain>
        struct ChooseBestMapping
        {
            // TODO
            using type = AoS<UserDomain, DatumDomain>;
        };
    }

    template<typename UserDomain, typename DatumDomain>
    using Auto =
        typename internal::ChooseBestMapping<UserDomain, DatumDomain>::type;
}
