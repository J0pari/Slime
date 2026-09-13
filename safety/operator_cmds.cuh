// Sheet S-002: Operator command parsing — pure host logic
//
// The line parser for operator_cmd.txt is kept free of CUDA dependencies so
// the production path (safety/alignment.cu) and the host unit tests execute
// the same function. The APPLIED semantics (pause gating, durable prune)
// live in the run loop; this header only classifies commands.

#ifndef COEVO_SAFETY_OPERATOR_CMDS_CUH
#define COEVO_SAFETY_OPERATOR_CMDS_CUH

#include "../config/constants.cuh"
#include "../config/strong_ids.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace slime::safety::alignment {

enum class OperatorCommand : uint8_t {
    None = 0,
    Pause,
    Resume,
    Checkpoint,
    Prune,
};

struct ParsedCommand {
    OperatorCommand command = OperatorCommand::None;
    LineageId lineage;   // set when command == Prune
};

// Parse one line of operator_cmd.txt (newline already stripped). Unknown
// lines parse to OperatorCommand::None.
inline ParsedCommand parse_operator_line(const char* line) {
    ParsedCommand out;
    if (std::strncmp(line, "prune ", 6) == 0) {
        out.command = OperatorCommand::Prune;
        out.lineage = LineageId(static_cast<uint32_t>(std::strtoul(line + 6, nullptr, 10)));
    } else if (std::strcmp(line, "pause") == 0) {
        out.command = OperatorCommand::Pause;
    } else if (std::strcmp(line, "resume") == 0) {
        out.command = OperatorCommand::Resume;
    } else if (std::strcmp(line, "checkpoint") == 0) {
        out.command = OperatorCommand::Checkpoint;
    }
    return out;
}

// Durable operator state owned by the run loop (not transient locals).
struct OperatorState {
    bool paused = false;
    bool checkpoint_requested = false;
    LineageId pruned_lineages[TOTAL_ORG];
    int n_pruned = 0;

    bool lineage_pruned(LineageId lineage) const {
        for (int i = 0; i < n_pruned; ++i) {
            if (pruned_lineages[i] == lineage) return true;
        }
        return false;
    }

    void add_pruned(LineageId lineage) {
        if (lineage_pruned(lineage)) return;
        if (n_pruned < TOTAL_ORG) pruned_lineages[n_pruned++] = lineage;
    }
};

}  // namespace slime::safety::alignment

#endif  // COEVO_SAFETY_OPERATOR_CMDS_CUH
