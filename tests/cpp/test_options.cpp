#include <gtest/gtest.h>
#include "pyflame_rt/options.hpp"

using namespace pyflame_rt;

TEST(SessionOptionsTest, Defaults) {
    SessionOptions opts;

    EXPECT_EQ(opts.device, "cpu");
    EXPECT_EQ(opts.num_threads, 0);
    EXPECT_FALSE(opts.enable_profiling);
    EXPECT_EQ(opts.execution_mode, "sequential");
    EXPECT_EQ(opts.log_level, "warning");
}

TEST(SessionOptionsTest, ValidOptions) {
    SessionOptions opts;
    opts.device = "cpu";
    opts.num_threads = 4;
    opts.log_level = "info";

    auto errors = opts.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(SessionOptionsTest, InvalidDevice) {
    SessionOptions opts;
    opts.device = "invalid";

    auto errors = opts.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_EQ(errors.size(), 1);
}

TEST(SessionOptionsTest, InvalidLogLevel) {
    SessionOptions opts;
    opts.log_level = "trace";

    auto errors = opts.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(SessionOptionsTest, InvalidExecutionMode) {
    SessionOptions opts;
    opts.execution_mode = "async";

    auto errors = opts.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(SessionOptionsTest, NegativeThreads) {
    SessionOptions opts;
    opts.num_threads = -1;

    auto errors = opts.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(RunOptionsTest, Defaults) {
    RunOptions opts;

    EXPECT_FALSE(opts.log_level.has_value());
    EXPECT_FALSE(opts.tag.has_value());
    EXPECT_FALSE(opts.timeout_ms.has_value());
}

TEST(CompileOptionsTest, Defaults) {
    CompileOptions opts;

    EXPECT_FALSE(opts.cache_dir.has_value());
    EXPECT_FALSE(opts.dynamic_batch);
    EXPECT_EQ(opts.optimization_level, 2);
    EXPECT_TRUE(opts.input_shapes.empty());
}
