#include "../tools/ComputePipelineFixture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckResult(const AgcDriver::ComputePipelineReplayResult& result) {
    Require(!result.deviceName.empty() && result.vendorId != 0 && result.apiVersion != 0, "replay did not report its device");
    Require(result.cache.size() >= 32, "replay did not export a pipeline cache header");
    for (const auto duration : {result.moduleMilliseconds, result.layoutMilliseconds, result.pipelineMilliseconds, result.cacheMilliseconds, result.totalMilliseconds}) {
        Require(std::isfinite(duration) && duration >= 0, "replay reported an invalid duration");
    }
    Require(result.totalMilliseconds >= result.pipelineMilliseconds, "pipeline duration exceeds replay duration");
}

}

int main() {
    try {
        auto capture = AgcDriver::ComputePipelineFixture();
        const auto encoded = capture.Encode();
        const auto decoded = AgcDriver::ComputePipelineCapture::Decode(encoded);
        std::unique_ptr<AgcDriver::VulkanDevice> device;
        try {
            device = std::make_unique<AgcDriver::VulkanDevice>(AgcDriver::VulkanDevice::CompileOnly{});
        } catch (const std::exception& error) {
            const std::string_view message(error.what());
            const bool absent = (message.starts_with("Vulkan loader: ") && message != "Vulkan loader: vkGetInstanceProcAddr missing") || message == "Vulkan: no Vulkan 1.1 graphics and compute queue" || message == "vkCreateInstance: Vulkan result -9";
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr || !absent) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return 77;
        }
        const auto fresh = device->ReplayComputePipeline(decoded);
        CheckResult(fresh);
        Require(fresh.initialCacheBytes == 0, "fresh replay imported cache bytes");
        const auto imported = device->ReplayComputePipeline(decoded, fresh.cache);
        CheckResult(imported);
        Require(imported.initialCacheBytes == fresh.cache.size(), "replay did not import the supplied cache");
        Require(fresh.deviceId == imported.deviceId && fresh.driverVersion == imported.driverVersion && fresh.pipelineCacheUuid == imported.pipelineCacheUuid, "replay device identity changed");
        Require(capture.Encode() == encoded, "replay modified the captured pipeline");
        auto corrupt = fresh.cache;
        corrupt[8] ^= std::byte{0xff};
        bool rejected = false;
        try {
            device->ReplayComputePipeline(decoded, corrupt);
        } catch (const std::exception&) {
            rejected = true;
        }
        Require(rejected, "cache from another vendor was accepted");
        corrupt = fresh.cache;
        corrupt[16] ^= std::byte{0xff};
        rejected = false;
        try {
            device->ReplayComputePipeline(decoded, corrupt);
        } catch (const std::exception&) {
            rejected = true;
        }
        Require(rejected, "cache with another pipeline UUID was accepted");
        capture.requiredSubgroupSize = 32;
        try {
            const auto subgroup = device->ReplayComputePipeline(capture);
            CheckResult(subgroup);
            std::puts("compute pipeline replay: required subgroup size 32 compiled");
        } catch (const std::exception& error) {
            Require(std::string_view(error.what()).find("captured required subgroup size is unsupported") != std::string_view::npos, "required subgroup replay failed without an explicit support refusal");
            std::puts("compute pipeline replay: device explicitly refused required subgroup size 32");
        }
        std::printf("compute pipeline replay: fresh %.6f ms, imported %.6f ms, GPU %s; no dispatch\n", fresh.pipelineMilliseconds, imported.pipelineMilliseconds, fresh.deviceName.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "compute pipeline replay test: %s\n", error.what());
        return 1;
    }
}
