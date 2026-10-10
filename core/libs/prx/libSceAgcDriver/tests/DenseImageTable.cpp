#include "Optimization/DenseImageTable.hpp"
#include <cstdio>
#include <stdexcept>

namespace {

using ShaderRecompiler::BuildDenseImageTable;
using ShaderRecompiler::DenseImageTable;

void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void LargeSharedHeap() {
    std::vector<DenseImageTable::Descriptor> entries(18960u);
    constexpr std::uint32_t nulls = 2565u;
    constexpr std::uint32_t unique = 270u;
    for (std::uint32_t key = nulls; key < entries.size(); ++key) {
        auto& descriptor = entries[key];
        descriptor[0] = 1u + (key - nulls) % unique;
        descriptor[1] = 56u << 20u;
        descriptor[3] = (9u << 28u) | 0xfacu;
    }
    const auto table = BuildDenseImageTable(entries, 65536u);
    Require(table.descriptors.size() == unique, "wide table did not compact repeated descriptors");
    Require(table.keys.size() == entries.size(), "wide table dropped the high key range");
    for (std::uint32_t key = 0u; key < entries.size(); ++key) {
        const auto slot = table.Lookup(key);
        if (key < nulls) Require(slot == DenseImageTable::Unmapped, "null entry acquired a descriptor");
        else {
            Require(slot < table.descriptors.size(), "valid key has no pool entry");
            Require(table.descriptors[slot] == entries[key], "high-key descriptor bits changed");
        }
    }
    Require(table.Lookup(18959u) != DenseImageTable::Unmapped, "last table key is unmapped");
    Require(table.Lookup(18960u) == DenseImageTable::Unmapped, "past-end key is mapped");
    Require(table.Lookup(DenseImageTable::Unmapped) == DenseImageTable::Unmapped, "UINT_MAX key is mapped");
    const auto again = BuildDenseImageTable(entries, 65536u);
    Require(table.descriptors == again.descriptors && table.keys == again.keys, "pool construction is not deterministic");
}

void DescriptorIdentity() {
    DenseImageTable::Descriptor base{1u, 56u << 20u, 0u, (9u << 28u) | 0xfacu, 0u, 0u, 0u, 0u};
    std::vector<DenseImageTable::Descriptor> entries{base};
    for (std::uint32_t word = 0u; word < base.size(); ++word) {
        auto changed = base;
        changed[word] ^= 0x100u;
        entries.push_back(changed);
    }
    entries.push_back(base);
    const auto table = BuildDenseImageTable(entries, 65536u);
    Require(table.descriptors.size() == 9u, "deduplication ignored a descriptor word");
    Require(table.Lookup(0u) == table.Lookup(9u), "identical descriptors occupy different slots");
    DenseImageTable::Descriptor nullDescriptor{};
    nullDescriptor[1] = 56u << 20u;
    nullDescriptor[3] = 9u << 28u;
    const std::array<DenseImageTable::Descriptor, 1> nulls{nullDescriptor};
    const auto empty = BuildDenseImageTable(nulls, 1u);
    Require(empty.descriptors.empty() && empty.Lookup(0u) == DenseImageTable::Unmapped, "null address with nonzero format was bound");
    Require(BuildDenseImageTable({}, 0u).keys.empty(), "empty table is not empty");
    bool rejected = false;
    try { static_cast<void>(BuildDenseImageTable(entries, 9u)); }
    catch (const std::runtime_error&) { rejected = true; }
    Require(rejected, "entry limit did not reject an oversized table");
}

}

int main() {
    try {
        LargeSharedHeap();
        DescriptorIdentity();
        std::puts("dense image table CPU tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
