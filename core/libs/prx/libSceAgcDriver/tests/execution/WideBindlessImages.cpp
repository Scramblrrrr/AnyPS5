#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

namespace {

using AgcDriver::Graphics::Require;
using namespace ShaderRecompiler;

struct alignas(4096) GuestData {
    std::array<std::array<std::uint32_t, 8>, 18960> heap{};
    std::array<std::uint32_t, 800> material{};
    std::array<std::uint32_t, 16> srt{};
    std::array<std::uint32_t, 64> output{};
};

alignas(256) std::array<std::array<std::uint32_t, 64>, 270> textures{};
std::array<GuestData, 5> guest;
constexpr std::array<std::uint32_t, 23> code{0xbf068002u, 0xbf850013u, 0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u, 0x7e200500u, 0x93109010u, 0xf4200406u, 0x20000004u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xf09c0f08u, 0x00450000u, 0xe0700000u, 0x80070000u, 0xbf810000u, 0xbf800000u};

std::array<std::uint32_t, 4> Buffer(const void* pointer, std::uint32_t stride, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

void Run(AgcDriver::VulkanDevice& device, bool fault) {
    for (std::uint32_t texture=0; texture < textures.size(); ++texture) textures[texture].fill(0x40000000u + texture);
    const auto threads = std::min(32u, device.Target().subgroupSize);
    CompiledShaderArtifact artifact;
    for (std::uint32_t iteration = 0u; iteration < guest.size(); ++iteration) {
        auto& data = guest[iteration];
        const auto table = Buffer(data.heap.data(), 32u, static_cast<std::uint32_t>(data.heap.size()));
        const auto material = Buffer(data.material.data(), 16u, 200u);
        std::copy(table.begin(), table.end(), data.srt.begin());
        std::copy(material.begin(), material.end(), data.srt.begin() + 8u);
        const auto srtAddress = reinterpret_cast<std::uintptr_t>(data.srt.data());
        const std::array<std::uint32_t, 3> userData{static_cast<std::uint32_t>(srtAddress), static_cast<std::uint32_t>(srtAddress >> 32u), iteration == 4u ? 0u : 1u};
        const std::array regions{MemoryRegion{reinterpret_cast<std::uintptr_t>(&data), std::as_bytes(std::span(&data, 1u))}};
        for (std::uint32_t key = 0; key < data.heap.size(); ++key) {
            const auto texture = key % textures.size();
            const auto address = reinterpret_cast<std::uintptr_t>(textures[texture].data());
            data.heap[key] = {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u) | (20u << 20u) | (3u << 30u), 7u, 0x90000facu, 0u, 0u, 0u, 0u};
        }
        for (std::uint32_t record=0; record < 200u; ++record) data.material[record * 4u + 1u] = record;
        const std::array<std::uint32_t, 5> selected{18959u, 18959u - 270u, 18958u, 0xffffffffu, 18959u};
        data.material[1] = selected[iteration];
        data.heap[18958u] = {};
        data.heap[17u][3] &= 0x0fffffffu;
        if (fault) data.material[1] = 17u;
        data.output.fill(0xdeadbeefu);
        const auto output = Buffer(data.output.data(), 0u, 256u);
        std::copy(output.begin(), output.end(), data.srt.begin() + 12u);
        const ShaderComputeStageInfo compute{{threads, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0u, {}}, {32u, 0u, userData, compute, std::nullopt, std::nullopt, regions}, device.Target(), {0u, 0u, 0u, 128u}};
        const auto shader = Recompile(request);
        if (artifact.variantId == 0u) artifact = shader;
        else Require(shader.cacheHit && shader.variantId == artifact.variantId, "bindless table changed the compiled artifact");
        const auto heap = std::ranges::find_if(shader.bindings, [](const DescriptorBinding& value) { return value.role == DescriptorRole::GuestImages && value.count > 128u; });
        if (iteration != 4u) Require(heap != shader.bindings.end(), "wide GPU fixture did not bind more than 128 images");
        if (iteration == 0u && !fault) {
            auto invalid = shader;
            const auto flat = std::ranges::find_if(invalid.bindings, [](const DescriptorBinding& value) { return value.role == DescriptorRole::FlattenedSrt; });
            Require(flat != invalid.bindings.end() && invalid.runtimeImageTableOffsets.size() == 1u, "wide fixture lost its table header");
            flat->guestDescriptor.at(invalid.runtimeImageTableOffsets.front() + 1u) = 0xffffffffu;
            bool rejected = false;
            try {
                device.Dispatch(invalid, 1u, 1u, 1u);
            } catch (const std::exception& error) {
                rejected = std::string_view(error.what()).find("dense image key map exceeds flattened SRT") != std::string_view::npos;
            }
            Require(rejected, "driver accepted an invalid dense key-map offset");
        }
        device.Dispatch(shader, 1u, 1u, 1u);
        device.SubmitRecorded(false);
        if (fault) {
            device.WaitIdle();
            Require(data.output[0] == 0xdeadbeefu, "invalid wide-table access executed the following output store");
            return;
        }
    }
    device.WaitIdle();
    for (std::uint32_t iteration = 0u; iteration < guest.size(); ++iteration) {
        const auto expected = iteration == 4u ? 0xdeadbeefu : iteration < 2u ? textures[18959u % textures.size()][0] : 0u;
        Require(guest[iteration].output[0] == expected, "bindless resource selection or lifetime failed at " + std::to_string(iteration) + " value=" + std::to_string(guest[iteration].output[0]) + " expected=" + std::to_string(expected));
    }
}

}

int main(int argc, char** argv) {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, argc == 2 && std::string_view(argv[1]) == "--fault");
        std::cout << "wide bindless image execution tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
