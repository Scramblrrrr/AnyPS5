#include "../tools/ComputePipelineFixture.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename TAction>
void Reject(TAction action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    Require(rejected, message);
}

class Directory {
public:
    Directory() {
        path = std::filesystem::temp_directory_path() / ("anyps5-compute-capture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Require(std::filesystem::create_directory(path), "could not create test directory");
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

void CheckRoundTrip() {
    auto capture = AgcDriver::ComputePipelineFixture();
    capture.deviceName = "test GPU \"name\"";
    capture.vendorId = 0x10de;
    capture.deviceId = 0x2b85;
    capture.driverVersion = 0x12345678;
    capture.apiVersion = VK_API_VERSION_1_3;
    capture.pipelineCacheUuid[0] = 0x12;
    capture.pipelineCacheUuid[15] = 0xab;
    capture.requiredSubgroupSize = 32;
    capture.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
    capture.sets = {{VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                     {{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
                      {7, VK_DESCRIPTOR_TYPE_SAMPLER, 2, VK_SHADER_STAGE_COMPUTE_BIT, 0}}}};
    capture.pushConstants = {{VK_SHADER_STAGE_COMPUTE_BIT, 16, 32}};
    const auto encoded = capture.Encode();
    const auto decoded = AgcDriver::ComputePipelineCapture::Decode(encoded);
    Require(decoded.Encode() == encoded, "capture round trip changed pipeline inputs");
    Require(decoded.Identity() == capture.Identity(), "capture identity changed after decoding");
    auto changed = decoded;
    ++changed.specialization[0].value;
    Require(changed.Identity() != capture.Identity(), "specialization did not affect capture identity");
    changed = decoded;
    ++changed.sets[0].bindings[0].count;
    Require(changed.Identity() != capture.Identity(), "descriptor layout did not affect capture identity");
    changed = decoded;
    changed.requiredSubgroupSize = 64;
    Require(changed.Identity() != capture.Identity(), "subgroup size did not affect capture identity");
    Directory directory;
    const auto file = capture.Write(directory.path);
    Require(AgcDriver::ComputePipelineCapture::Read(file).Encode() == encoded, "capture file changed pipeline inputs");
    Require(capture.Write(directory.path) == file, "identical capture was not deduplicated");
    std::ofstream(file, std::ios::binary | std::ios::trunc).put('x');
    Reject([&] { capture.Write(directory.path); }, "existing damaged capture was overwritten");
    Require(std::filesystem::file_size(file) == 1, "damaged capture was changed");
}

void CheckMalformedRecords() {
    const auto capture = AgcDriver::ComputePipelineFixture();
    const auto encoded = capture.Encode();
    for (std::size_t length = 0; length < encoded.size(); ++length) {
        Reject([&] { AgcDriver::ComputePipelineCapture::Decode(std::span(encoded).first(length)); }, "truncated capture accepted");
    }
    auto bytes = encoded;
    bytes.push_back(std::byte{0});
    Reject([&] { AgcDriver::ComputePipelineCapture::Decode(bytes); }, "trailing capture bytes accepted");
    bytes = encoded;
    bytes[0] ^= std::byte{1};
    Reject([&] { AgcDriver::ComputePipelineCapture::Decode(bytes); }, "invalid signature accepted");
    bytes = encoded;
    bytes[8] = std::byte{0xff};
    Reject([&] { AgcDriver::ComputePipelineCapture::Decode(bytes); }, "unknown capture version accepted");
    bytes = encoded;
    for (std::size_t index = 104; index < 108; ++index) bytes[index] = std::byte{0xff};
    Reject([&] { AgcDriver::ComputePipelineCapture::Decode(bytes); }, "unbounded device name accepted");
    const auto invalid = [&](auto change) {
        auto copy = capture;
        change(copy);
        Reject([&] { copy.Encode(); }, "unsupported pipeline input accepted");
    };
    invalid([](auto& copy) { copy.spirv[0] = 0; });
    invalid([](auto& copy) { copy.spirv[5] = 0; });
    invalid([](auto& copy) { copy.spirv[11] = 4; });
    invalid([](auto& copy) { copy.spirv[13] = 0; });
    invalid([](auto& copy) { copy.flags = VK_PIPELINE_CREATE_DERIVATIVE_BIT; });
    invalid([](auto& copy) { copy.stageFlags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT; });
    invalid([](auto& copy) { copy.requiredSubgroupSize = 3; });
    invalid([](auto& copy) { copy.specialization.push_back(copy.specialization[0]); });
    invalid([](auto& copy) { copy.pushConstants = {{VK_SHADER_STAGE_VERTEX_BIT, 0, 4}}; });
    invalid([](auto& copy) { copy.pushConstants = {{VK_SHADER_STAGE_COMPUTE_BIT, 1, 4}}; });
    invalid([](auto& copy) { copy.sets = {{0, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, VK_SHADER_STAGE_COMPUTE_BIT, 0}}}}; });
    invalid([](auto& copy) { copy.sets = {{0, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0}}}}; });
    invalid([](auto& copy) { copy.sets = {{0, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT}}}}; });
    invalid([](auto& copy) { copy.sets = {{0, {{}, {}}}}; });
}

}

int main() {
    try {
        CheckRoundTrip();
        CheckMalformedRecords();
        std::puts("compute pipeline capture: round trip, immutable writes, truncated records and unsupported inputs passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "compute pipeline capture test: %s\n", error.what());
        return 1;
    }
}
