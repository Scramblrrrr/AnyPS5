#include "Recompiler.hpp"
#include "BdaAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

std::array<std::uint32_t, 64> Input{};
std::array<std::uint32_t, 64> Output{};

std::uint32_t HostLanes(std::initializer_list<std::uint32_t> body, std::uint32_t threads, std::uint32_t ldsDwords) {
    std::vector<std::uint32_t> code{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u};
    code.insert(code.end(), body);
    code.insert(code.end(), {0xe0701000u, 0x80010201u, 0xbf810000u});
    const auto inputAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Input.data()));
    const auto outputAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Output.data()));
    const std::array<std::uint32_t, 8> userData{
        static_cast<std::uint32_t>(inputAddress), static_cast<std::uint32_t>((inputAddress >> 32u) & 0xffffu), 256u, 0x31016facu,
        static_cast<std::uint32_t>(outputAddress), static_cast<std::uint32_t>((outputAddress >> 32u) & 0xffffu), 256u, 0x31016facu};
    static constexpr std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    static constexpr std::array<std::uint32_t, 8> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityGroupNonUniform,
        spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformArithmetic};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}};
    request.context.waveSize = 64;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{threads, 1u, 1u}, ldsDwords, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00401000u;
    request.target.spirvVersion = 0x00010300u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = BdaAbi::Version;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.fragmentShaderBarycentricEnabled = false;
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    return Recompile(request).hostSubgroupSize;
}

void Expect(const char* name, std::uint32_t lanes, std::uint32_t expected) {
    if (lanes != expected) throw std::runtime_error(std::string(name) + ": host subgroup size " + std::to_string(lanes) + ", expected " + std::to_string(expected));
}

}

int main() {
    try {
        Expect("per-lane arithmetic", HostLanes({0x4a040481u}, 64u, 0u), 64u);
        Expect("vcc read as a scalar", HostLanes({0x7d880085u, 0xbe88046au, 0x7e040208u}, 64u, 0u), 32u);
        Expect("v_readfirstlane of a lane value", HostLanes({0x7e100502u, 0x7e040208u}, 64u, 0u), 32u);
        Expect("execz region with a scalar result used after it", HostLanes({0x7d880085u, 0xbe8a246au, 0xbf880001u, 0x800c8100u, 0x88fe0a7eu, 0x7e04020cu}, 64u, 0u), 32u);
        Expect("execz region with vector work only", HostLanes({0x7d880085u, 0xbe8a246au, 0xbf880001u, 0x4a040481u, 0x88fe0a7eu}, 64u, 0u), 64u);
        Expect("LDS in a one-wave workgroup", HostLanes({0xd8340000u, 0x00000201u, 0xbf8cc07fu, 0xd8d80000u, 0x02000001u, 0xbf8cc07fu}, 64u, 64u), 64u);
        Expect("LDS writes in a two-wave workgroup", HostLanes({0xd8340000u, 0x00000201u, 0xbf8cc07fu, 0xd8d80000u, 0x02000001u, 0xbf8cc07fu}, 128u, 128u), 32u);
        std::puts("wave64 layout tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
