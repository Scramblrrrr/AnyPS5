#include "prx/libSceAgcDriver/Execution/include/HostAllocationProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/StackWalk.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <cstdlib>
#include <stdexcept>
#ifdef _WIN32
#include "mimalloc-profile.h"
#include <windows.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace {

constexpr std::size_t SampleRate = 8 * 1024 * 1024;
constexpr std::size_t EntryCount = 8192;
constexpr std::size_t Depth = 16;

struct Entry {
    std::array<std::uint64_t, Depth> frames{};
    unsigned count = 0;
    std::atomic<std::uint64_t> liveBytes{0};
    std::atomic<std::uint64_t> liveObjects{0};
    std::atomic<std::uint64_t> totalBytes{0};
    std::atomic<std::uint64_t> totalObjects{0};
};

struct State {
    mi_profiler_t profiler{};
    std::unique_ptr<AgcDriver::DwarfFrames> module;
    std::array<Entry, EntryCount> entries{};
    SRWLOCK lock = SRWLOCK_INIT;
    std::wstring output;
    std::atomic<std::uint64_t> nextSnapshot{0};
    std::atomic<std::uint64_t> dropped{0};
    unsigned sequence = 0;
    bool (*attach)(mi_profiler_t*) = nullptr;
    bool (*start)(mi_profiler_t*) = nullptr;
    bool (*stop)(mi_profiler_t*) = nullptr;
    void (*snapshot)(mi_profiler_t*) = nullptr;
};

State* state = nullptr;
std::once_flag initialized;
thread_local bool callbackActive = false;

unsigned Capture(std::array<std::uint64_t, Depth>& frames) {
    CONTEXT context{};
    RtlCaptureContext(&context);
    MEMORY_BASIC_INFORMATION stack{};
    if (VirtualQuery(reinterpret_cast<void*>(context.Rsp), &stack, sizeof(stack)) != sizeof(stack)) return 0;
    if (stack.State != MEM_COMMIT || (stack.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return 0;
    const auto low = reinterpret_cast<std::uint64_t>(stack.BaseAddress);
    const auto high = low + stack.RegionSize;
    unsigned count = 0;
    for (unsigned depth = 0; depth < 32 && context.Rip != 0 && count < frames.size(); ++depth) {
        if (context.Rsp < low || context.Rsp >= high) break;
        if (state->module->Contains(context.Rip)) frames[count++] = context.Rip - state->module->Base();
        DWORD64 imageBase = 0;
        if (auto* entry = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr)) {
            PVOID handler = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, entry, &context, &handler, &establisher, nullptr);
            continue;
        }
        if (!state->module->Contains(context.Rip - (depth == 0 ? 0 : 1))) break;
        AgcDriver::DwarfFrames::Registers registers{context.Rax, context.Rdx, context.Rcx, context.Rbx, context.Rsi, context.Rdi, context.Rbp, context.Rsp, context.R8, context.R9, context.R10, context.R11, context.R12, context.R13, context.R14, context.R15, context.Rip};
        const auto step = state->module->Unwind(registers, depth == 0, [&](std::uint64_t address, std::uint64_t& value) {
            if (address < low || address > high - sizeof(value)) return false;
            std::memcpy(&value, reinterpret_cast<void*>(address), sizeof(value));
            return true;
        });
        if (step != AgcDriver::DwarfFrames::Step::Stepped) break;
        DWORD64* columns[]{&context.Rax, &context.Rdx, &context.Rcx, &context.Rbx, &context.Rsi, &context.Rdi, &context.Rbp, &context.Rsp, &context.R8, &context.R9, &context.R10, &context.R11, &context.R12, &context.R13, &context.R14, &context.R15, &context.Rip};
        for (std::size_t column = 0; column < registers.size(); ++column) *columns[column] = registers[column];
    }
    return count;
}

void Snapshot(mi_profiler_t*) {
    const bool previous = std::exchange(callbackActive, true);
    struct Restore {
        bool value;
        ~Restore() { callbackActive = value; }
    } restore{previous};
    const auto destination = state->output + L"." + std::to_wstring(state->sequence++ % 4) + L".tsv";
    const auto temporary = destination + L".tmp";
    auto* file = _wfopen(temporary.c_str(), L"w");
    if (file == nullptr) return;
    std::fprintf(file, "schema=1 sample_rate=%llu tick=%llu dropped=%llu raw_sampled_counts=1 own_driver_frames_only=1\n", static_cast<unsigned long long>(SampleRate), static_cast<unsigned long long>(GetTickCount64()), static_cast<unsigned long long>(state->dropped.load()));
    std::fputs("live_bytes\tlive_objects\ttotal_bytes\ttotal_objects\tframes\n", file);
    AcquireSRWLockShared(&state->lock);
    for (const auto& entry : state->entries) {
        if (entry.count == 0) continue;
        std::fprintf(file, "%llu\t%llu\t%llu\t%llu\t", static_cast<unsigned long long>(entry.liveBytes.load()), static_cast<unsigned long long>(entry.liveObjects.load()), static_cast<unsigned long long>(entry.totalBytes.load()), static_cast<unsigned long long>(entry.totalObjects.load()));
        for (unsigned index = 0; index < entry.count; ++index) std::fprintf(file, "%s%llx", index == 0 ? "" : ";", static_cast<unsigned long long>(entry.frames[index]));
        std::fputc('\n', file);
    }
    ReleaseSRWLockShared(&state->lock);
    const bool written = std::ferror(file) == 0;
    const bool closed = std::fclose(file) == 0;
    if (written && closed) MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING);
}

std::size_t OnAllocate(mi_profiler_t*, mi_profiler_sample_data_t* data, void*, std::size_t bytes, std::size_t, std::uint64_t, mi_heap_t*) {
    data->user_data[0] = nullptr;
    data->user_data[1] = reinterpret_cast<void*>(bytes);
    if (callbackActive) return SampleRate;
    callbackActive = true;
    struct Restore {
        ~Restore() { callbackActive = false; }
    } restore;
    std::array<std::uint64_t, Depth> frames{};
    const auto count = Capture(frames);
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned index = 0; index < count; ++index) hash = (hash ^ frames[index]) * 1099511628211ull;
    Entry* found = nullptr;
    if (count != 0) {
        AcquireSRWLockExclusive(&state->lock);
        for (std::size_t probe = 0; probe < EntryCount; ++probe) {
            auto& entry = state->entries[(hash + probe) % EntryCount];
            if (entry.count != 0 && (entry.count != count || entry.frames != frames)) continue;
            if (entry.count == 0) {
                entry.frames = frames;
                entry.count = count;
            }
            found = &entry;
            break;
        }
        ReleaseSRWLockExclusive(&state->lock);
    }
    if (found != nullptr) {
        found->liveBytes.fetch_add(bytes);
        found->liveObjects.fetch_add(1);
        found->totalBytes.fetch_add(bytes);
        found->totalObjects.fetch_add(1);
        data->user_data[0] = found;
    } else {
        state->dropped.fetch_add(1);
    }
    const auto now = GetTickCount64();
    auto next = state->nextSnapshot.load();
    if (now >= next && state->nextSnapshot.compare_exchange_strong(next, now + 15000)) state->snapshot(&state->profiler);
    return SampleRate;
}

void OnFree(mi_profiler_t*, mi_profiler_sample_data_t* data, void*, mi_heap_t*) {
    auto* entry = static_cast<Entry*>(data->user_data[0]);
    if (entry == nullptr) return;
    const auto address = reinterpret_cast<std::uintptr_t>(entry);
    const auto begin = reinterpret_cast<std::uintptr_t>(state->entries.data());
    if (address < begin || address >= begin + sizeof(state->entries) || (address - begin) % sizeof(Entry) != 0) return;
    entry->liveBytes.fetch_sub(reinterpret_cast<std::size_t>(data->user_data[1]));
    entry->liveObjects.fetch_sub(1);
}

void Stop() {
    state->stop(&state->profiler);
    state->attach(nullptr);
    state->snapshot(&state->profiler);
}

void Initialize(const char* configured) {
    if (std::getenv("MIMALLOC_PROFILE") != nullptr) throw std::runtime_error("native allocation profiling cannot share the built-in allocator profiler");
    const auto allocator = GetModuleHandleW(L"mimalloc.dll");
    if (allocator == nullptr) throw std::runtime_error("native allocation profiling requires the loaded mimalloc provider");
    auto* value = new State;
    value->attach = reinterpret_cast<decltype(value->attach)>(GetProcAddress(allocator, "mi_profile"));
    value->start = reinterpret_cast<decltype(value->start)>(GetProcAddress(allocator, "mi_profiler_start"));
    value->stop = reinterpret_cast<decltype(value->stop)>(GetProcAddress(allocator, "mi_profiler_stop"));
    value->snapshot = reinterpret_cast<decltype(value->snapshot)>(GetProcAddress(allocator, "mi_profiler_snapshot"));
    if (!value->attach || !value->start || !value->stop || !value->snapshot) throw std::runtime_error("mimalloc provider has no native profiling API");
    HMODULE own = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&Capture), &own)) throw std::runtime_error("native allocation profiler cannot identify its own module");
    const auto base = reinterpret_cast<std::uint64_t>(own);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    const AgcDriver::DwarfFrames::Byte* frames = nullptr;
    const AgcDriver::DwarfFrames::Byte* framesEnd = nullptr;
    for (unsigned index = 0; index < nt->FileHeader.NumberOfSections; ++index) {
        if (std::memcmp(sections[index].Name, ".ehfram", 8) != 0) continue;
        frames = reinterpret_cast<const AgcDriver::DwarfFrames::Byte*>(base + sections[index].VirtualAddress);
        framesEnd = frames + sections[index].Misc.VirtualSize;
    }
    if (frames == nullptr) throw std::runtime_error("native allocation profiler requires its own DWARF unwind table");
    value->module = std::make_unique<AgcDriver::DwarfFrames>(base, base + nt->OptionalHeader.SizeOfImage, frames, framesEnd);
    value->output = std::filesystem::path(reinterpret_cast<const char8_t*>(configured)).wstring();
    value->profiler.sample_data_size = 2 * sizeof(void*);
    value->profiler.initial_sample_rate = SampleRate;
    value->profiler.on_alloc = OnAllocate;
    value->profiler.on_free = OnFree;
    value->profiler.on_snapshot = Snapshot;
    value->nextSnapshot = GetTickCount64() + 15000;
    state = value;
    if (!state->attach(&state->profiler)) throw std::runtime_error("native allocation profiler cannot replace an existing profiler");
    try {
        LibcRegisterShutdown_nid_postfix(Stop);
    } catch (...) {
        state->attach(nullptr);
        throw;
    }
    state->start(&state->profiler);
}

}
#endif

void AgcDriver::StartHostAllocationProfile_nid_no_patch() {
    const auto* configured = std::getenv("APS5_PROFILE_HOST_ALLOCATIONS");
    if (configured == nullptr || *configured == 0) return;
#ifdef _WIN32
    std::call_once(initialized, [configured] { Initialize(configured); });
#else
    throw std::runtime_error("native allocation profiling requires Windows");
#endif
}
