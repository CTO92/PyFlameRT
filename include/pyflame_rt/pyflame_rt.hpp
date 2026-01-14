#pragma once

/// @file pyflame_rt.hpp
/// @brief Main include file for PyFlameRT library

// Core types
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/errors.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/node.hpp"
#include "pyflame_rt/graph.hpp"

// Registry and backend
#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/backend.hpp"

// Session
#include "pyflame_rt/options.hpp"
#include "pyflame_rt/session.hpp"

namespace pyflame_rt {

/// Library version
constexpr const char* VERSION = "0.1.0";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

} // namespace pyflame_rt
