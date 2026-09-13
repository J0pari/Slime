// Strong identifiers (fallback/default audit, class 6).
//
// The pipeline is id-heavy: lineage identities, pool slots, genome seeds,
// archive indices, and PT slots are all integers today and interchangeable
// at compile time, so a cross-wiring mistake (a slot where a lineage is
// expected) is invisible until it corrupts state. StrongId makes each id
// family a distinct type: construction is explicit, conversion back to the
// underlying integer requires value(), and there is no implicit conversion
// in either direction. The tags are empty structs; the wrapper costs
// nothing at runtime.
//
// Default construction is deliberately absent as a *valid* value: storage
// that has not been assigned an id holds kInvalid, which no assigned id can
// equal (ids are handed out by a counter that never reaches the sentinel).
// Absence is therefore representable and distinguishable, not silently a
// real id.

#ifndef COEVO_CONFIG_STRONG_IDS_CUH
#define COEVO_CONFIG_STRONG_IDS_CUH

#include <cstdint>
#include <type_traits>

namespace slime {

template <typename Tag, typename T>
struct StrongId {
    T raw;

    __host__ __device__ constexpr StrongId() : raw(static_cast<T>(-1)) {}

    __host__ __device__ constexpr explicit StrongId(T value) : raw(value) {}

    __host__ __device__ constexpr T value() const { return raw; }

    __host__ __device__ constexpr bool valid() const {
        return raw != static_cast<T>(-1);
    }

    __host__ __device__ constexpr bool operator==(StrongId o) const {
        return raw == o.raw;
    }
    __host__ __device__ constexpr bool operator!=(StrongId o) const {
        return raw != o.raw;
    }
    __host__ __device__ constexpr bool operator<(StrongId o) const {
        return raw < o.raw;
    }
    __host__ __device__ constexpr bool operator<=(StrongId o) const {
        return raw <= o.raw;
    }
    __host__ __device__ constexpr bool operator>(StrongId o) const {
        return raw > o.raw;
    }
    __host__ __device__ constexpr bool operator>=(StrongId o) const {
        return raw >= o.raw;
    }
};

struct LineageTag;
struct PoolSlotTag;
struct GenomeSeedTag;
struct ArchiveSlotTag;
struct PtSlotTag;

using LineageId  = StrongId<LineageTag, std::uint32_t>;
using PoolSlot   = StrongId<PoolSlotTag, int>;
using GenomeSeed = StrongId<GenomeSeedTag, std::uint32_t>;
using ArchiveSlot = StrongId<ArchiveSlotTag, int>;
using PtSlot     = StrongId<PtSlotTag, int>;

// A strong id is not its raw integer and a raw integer is not a strong id;
// both directions require an explicit act.
static_assert(!std::is_convertible_v<LineageId, std::uint32_t>,
              "LineageId must not decay to its raw type");
static_assert(!std::is_convertible_v<std::uint32_t, LineageId>,
              "a raw integer must not silently become a LineageId");
static_assert(!std::is_convertible_v<PoolSlot, LineageId>,
              "slot and lineage ids must not interchange");
static_assert(!std::is_convertible_v<LineageId, PoolSlot>,
              "lineage and slot ids must not interchange");
static_assert(!std::is_convertible_v<GenomeSeed, LineageId>,
              "genome seeds and lineage ids must not interchange");
static_assert(std::is_trivially_copyable_v<LineageId>,
              "strong ids must stay trivially copyable for device buffers");

}  // namespace slime

#endif  // COEVO_CONFIG_STRONG_IDS_CUH
