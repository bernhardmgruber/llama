#include <experimental/meta>
#include <experimental/compiler>
#include <iostream>
#include <utility>
#include <vector>
#include <array>
#include <llama/llama.hpp>
#include "../common/demangle.hpp"

// LLAMA stubs for compiler explorer:
//namespace llama {
//    template <typename T, size_t S>
//    struct DA {};
//
//    template <typename Name, typename T>
//    struct DE {};
//
//    template <typename... T>
//    struct DS {};
//}

// domain declaration as we would in C++
namespace domain {
    struct Pos {
        float x;
        float y;
        float z;
    };

    // leave this for now since it generates duplicated tags
    // struct Momentum {
    //     float z;
    //     float x;
    // };

    struct Name {
        Pos pos;
        // Momentum mom;
        int weight;
        bool options[4];
    }; 
}

// TODO: the following needs to be generated
// namespace st {
//    struct Pos {};
//    struct X {};
//    struct Y {};
//    struct Z {};
//    struct Momentum {};
//    struct Weight {};
//    struct Options {};

//    using Name = llama::DS<
//        llama::DE< st::Pos, llama::DS<
//            llama::DE< st::X, float >,
//            llama::DE< st::Y, float >,
//            llama::DE< st::Z, float >
//        > >,
//        llama::DE< st::Momentum,llama::DS<
//            llama::DE< st::Z, double >,
//            llama::DE< st::X, double >
//        > >,
//        llama::DE< st::Weight, int >,
//        llama::DE< st::Options, llama::DA< bool, 4 > >
//    >;
// }

// llama extension
namespace llama::reflect {
    struct tag_source{};

    // TODO replace by std::vector once it's constexpr
    template <typename T, std::size_t Capacity = 100>
    struct vector {
        std::array<T, Capacity> data{};
        int size = 0;

        constexpr auto begin() const {
            return data.begin();
        }

        constexpr auto end() const {
            return data.begin() + size;
        }

        constexpr auto operator[](int i) const {
            return data[i];
        }

        constexpr void push_back(T t) {
            data[size] = t;
            size++;
        }
    };

    using namespace std::experimental;

    // FIXME: all using declarations in the namespace fragments should actually be just types

    consteval auto fundamentalDataMember(meta::info type) -> meta::info {
        return fragment namespace {
            using T = typename(%{type}); // ISSUE: it seems we cannot create a fragment that returns a type
        };
    }

    consteval auto arrayMember(meta::info type) -> meta::info {
        // QUESTION: can I get the extend and base type of an array from a meta::info?
        return fragment namespace {
            using T = llama::DA<std::remove_all_extents_t<typename(%{type})>, std::extent_v<typename(%{type})>>; // ISSUE: it seems we cannot create a fragment that returns a type
        };
    }

    consteval auto structMember(meta::info s) -> meta::info {
        vector<meta::info> elements;
        for (meta::info member : meta::data_member_range(s)) {
            const auto type = meta::type_of(member);

            meta::info arg;
            if (meta::is_class_type(type)) {
                arg = structMember(type);
            } else if (meta::is_array_type(type)) {
                arg = arrayMember(type);
            } else if (meta::is_fundamental_type(type)) {
                arg = fundamentalDataMember(type);
            } else {
                __reflect_print("Unsupported type/member: ", meta::name_of(type), meta::name_of(member));
                __compiler_error("");
            }
            
            // inject the tag type as a renamed copy, as suggested by Wyatt Childers
            meta::info tag = meta::definition_of(reflexpr(tag_source));
            meta::set_new_name(tag, meta::name_of(member));
            -> tag;
            // -> fragment namespace {
            //     // struct unqualid(meta::name_of(%{member})) {}; // BUG? fails for now
            // };

            // FIXME: element should actually be a meta::info carring a type
            auto element = fragment namespace {
                using T = llama::DE<typename(meta::name_of(%{member})), typename(%{arg})>; // FIXME: arg is probably also wrong, since it is not a reflection of a type
            };
            elements.push_back(element);
        }

        return fragment namespace {
            using T = llama::DS<
                typename(... %{elements}) // ERROR currently fails, because the meta::infos inside elements are not types, but fragments
            >;
        };
    }

    template <typename Struct>
    consteval void gen() {
        auto ds = structMember(reflexpr(Struct));
        -> ds;
    }
}

// trigger generation
namespace st {
    consteval {
        llama::reflect::gen<domain::Name>();
    }
}

// check for generated tags
consteval {
    using namespace std::experimental;
    for (meta::info member : meta::member_range(reflexpr(st)))
        __reflect_pretty_print(member);
}

int main(int argc, char* argv[]) {
    return 0;
}