#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<unsigned> events;
bool refuse = false;
std::uint64_t sourceAddress = 0;

void Guard(std::uint64_t address, std::size_t bytes) {
    if (address != sourceAddress || bytes != 16) throw std::runtime_error("guard received the wrong read range");
    events.push_back(1);
    if (refuse) throw std::runtime_error("synthetic read refused");
}

void Flush(std::uint64_t address, std::size_t bytes) {
    if (address != sourceAddress || bytes != 16) throw std::runtime_error("flush received the wrong read range");
    events.push_back(2);
}

void Run() {
    using namespace AgcDriver::GuestMemory;
    alignas(16) std::array<std::byte, 16> source{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::byte>(i);
    sourceAddress = reinterpret_cast<std::uint64_t>(source.data());
    SetReadGuard(Guard);
    SetFlushHook(Flush);
    struct Restore {
        ~Restore() {
            SetReadGuard(nullptr);
            SetFlushHook(nullptr);
        }
    } restore;
    for (unsigned mode = 0; mode < 3; ++mode) {
        const auto read = [&](std::span<std::byte> destination) {
            if (mode == 0) Read(sourceAddress, destination);
            else if (mode == 1) ReadCommitted(sourceAddress, destination);
            else ReadCommittedUnsynced(sourceAddress, destination);
        };
        for (bool rejected : {false, true}) {
            std::array<std::byte, 16> destination{};
            destination.fill(std::byte{0xa5});
            events.clear();
            refuse = rejected;
            bool threw = false;
            try {
                read(destination);
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()) != "synthetic read refused") throw;
                threw = true;
            }
            const auto expected = rejected || mode == 2 ? std::vector<unsigned>{1} : std::vector<unsigned>{1, 2};
            if (threw != rejected || events != expected) throw std::runtime_error("a read bypassed its guard or flushed before the guard");
            if (rejected) {
                if (!std::all_of(destination.begin(), destination.end(), [](std::byte value) { return value == std::byte{0xa5}; })) throw std::runtime_error("a refused read changed its destination");
            } else if (destination != source) throw std::runtime_error("an allowed read copied the wrong bytes");
        }
        events.clear();
        refuse = true;
        read({});
        if (!events.empty()) throw std::runtime_error("an empty read invoked a guard or flush");
    }
}

}

int main() {
    try {
        Run();
        std::puts("committed read guard tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
