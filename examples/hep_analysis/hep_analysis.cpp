#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct JetFlags{};
    struct Pt{};
    struct Eta{};
    struct JetIdx{};
    struct Charge{};
    struct Mass{};
    struct RunNumber{};
    struct EventId{};
    struct Electrons{};
    struct Muons{};
    struct Jets{};
    struct HLTTriggerMuon{};
}

using Jet = llama::DS<
    llama::DE<tag::JetFlags, int>,
    llama::DE<tag::Pt, float>,
    llama::DE<tag::Eta, float>
>;

using Electron = llama::DS<
    llama::DE<tag::JetIdx, int>, 
    llama::DE<tag::Charge, int>,
    llama::DE<tag::Pt, float>,
    llama::DE<tag::Eta, float>,
    llama::DE<tag::Mass, float>
>;

using Muon = llama::DS<
    llama::DE<tag::Charge, int>,
    llama::DE<tag::Pt, float>,
    llama::DE<tag::Eta, float>,
    llama::DE<tag::Mass, float>
    // + 30 more floats
>;

using Event = llama::DS<
    llama::DE<tag::RunNumber, int>,
    llama::DE<tag::EventId, int>,
    llama::DE<tag::Electrons, Electron[]>,
    llama::DE<tag::Muons, Muon[]>,
    llama::DE<tag::Jets, Jet[]>,
    llama::DE<tag::HLTTriggerMuon, bool>
>;
// clang-format on

int main()
{
    constexpr auto n = 15;
    constexpr auto arrayDomain = llama::ArrayDomain{n};
    constexpr auto mapping = llama::mapping::SoA{arrayDomain, Event{}};
    //auto view = allocView(mapping);
    std::ofstream{"hep_analysis.svg"} << llama::toSvg(mapping);
    std::ofstream{"hep_analysis.html"} << llama::toHtml(mapping);
}
