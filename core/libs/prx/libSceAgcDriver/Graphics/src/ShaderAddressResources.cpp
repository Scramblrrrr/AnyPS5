#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace AgcDriver::Graphics {

bool ShaderResources::StageChecksFaults(const ShaderRecompiler::RecompileResult& program) {
    static const bool loopGuarded = std::getenv("APS5_LOOP_GUARD") != nullptr;
    bool tables = false;
    bool faults = false;
    bool tableFaultsOnly = false;
    for (const auto& binding : program.bindings) {
        tables = tables || binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable;
        faults = faults || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
        tableFaultsOnly = tableFaultsOnly || binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap;
    }
    return tables || (faults && (!tableFaultsOnly || program.imageTableFaults != 0 || program.poisonedSrtReads != 0 || loopGuarded));
}

void ShaderResources::prepareAddressBindings(std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots) {
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        std::uint32_t tables = 0;
        std::uint32_t faults = 0;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer) continue;
            Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && binding.guestDescriptor.empty() && !binding.readOnly, "invalid BDA descriptor contract");
            if (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable) ++tables;
            else ++faults;
        }
        const bool faultOnly = tables == 0 && faults == 1 && shader.program->bdaAbiVersion == ShaderRecompiler::BdaAbi::Version;
        Require((tables == faults || faultOnly) && tables <= 1 && faults <= 1, "invalid BDA table and fault descriptors");
        Require(shader.program->bdaAbiVersion == (faults == 0 ? 0u : ShaderRecompiler::BdaAbi::Version), "incompatible BDA ABI version");
        // Rect-list validation needs a fault buffer, but never accesses guest addresses.
        usesBda = usesBda || tables != 0;
        usesFaultBuffer = usesFaultBuffer || faults != 0;
        faultChecks = faultChecks || StageChecksFaults(*shader.program);
        const bool tableMap = std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap; });
        untabledFaults = untabledFaults || (faults != 0 && !tableMap);
    }
    if (usesBda) {
        Require(context.bufferDeviceAddress, "buffer device address is not enabled");
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        guestMemory.AcquireRegistered();
        const auto snapshotsStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (const auto& snapshot : snapshots) guestMemory.AddSnapshot(snapshot);
        if (profile) GuestBufferMemory::CountAddressBuild(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - snapshotsStart).count());
    }
}

VkDescriptorBufferInfo ShaderResources::descriptor(Allocation& allocation) {
    if (allocation.guest) return guestMemory.Descriptor(allocation.address, allocation.size, allocation.adjustment);
    if (allocation.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
        Require(context.emptyBuffer != VK_NULL_HANDLE, "no placeholder buffer for a null V#");
        return {context.emptyBuffer, 0, allocation.size};
    }
    if (allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable || allocation.role == ShaderRecompiler::DescriptorRole::FaultBuffer) {
        Require(bda != nullptr, "BDA descriptors have no memory owner");
        return allocation.role == ShaderRecompiler::DescriptorRole::BdaPagetable ? bda->Table() : bda->Fault();
    }
    Require(allocation.buffer != nullptr, "shader data has no buffer owner");
    return {allocation.buffer->Handle(), 0, allocation.size};
}

}
