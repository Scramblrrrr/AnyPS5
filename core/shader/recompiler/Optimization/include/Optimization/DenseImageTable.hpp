#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DENSEIMAGETABLE_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DENSEIMAGETABLE_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

namespace ShaderRecompiler {

struct DenseImageTable {
    using Descriptor = std::array<std::uint32_t, 8>;
    static constexpr std::uint32_t Unmapped = std::numeric_limits<std::uint32_t>::max();
    std::vector<Descriptor> descriptors;
    std::vector<std::uint32_t> keys;

    [[nodiscard]] std::uint32_t Lookup(std::uint32_t key) const {
        return key < keys.size() ? keys[key] : Unmapped;
    }
};

inline DenseImageTable BuildDenseImageTable(std::span<const DenseImageTable::Descriptor> entries, std::uint32_t entryLimit) {
    if (entries.size() > entryLimit) throw std::runtime_error("bindless dense table exceeds its entry limit");
    DenseImageTable result;
    result.keys.reserve(entries.size());
    std::map<DenseImageTable::Descriptor, std::uint32_t> slots;
    for (const auto& descriptor : entries) {
        if (descriptor[0] == 0u && (descriptor[1] & 0xffu) == 0u) {
            result.keys.push_back(DenseImageTable::Unmapped);
            continue;
        }
        const auto next = static_cast<std::uint32_t>(result.descriptors.size());
        const auto [position, inserted] = slots.emplace(descriptor, next);
        if (inserted) result.descriptors.push_back(descriptor);
        result.keys.push_back(position->second);
    }
    return result;
}

}

#endif
