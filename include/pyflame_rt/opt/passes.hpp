#pragma once

/// @file passes.hpp
/// @brief Master include file for all optimization passes

#include "pyflame_rt/opt/pass.hpp"
#include "pyflame_rt/opt/pattern_matcher.hpp"
#include "pyflame_rt/opt/constant_folding.hpp"
#include "pyflame_rt/opt/dead_code_elimination.hpp"
#include "pyflame_rt/opt/cse.hpp"
#include "pyflame_rt/opt/operator_fusion.hpp"
#include "pyflame_rt/opt/layout_optimization.hpp"
