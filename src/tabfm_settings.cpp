#include "tabfm_registration.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

#include <thread>

namespace duckdb {
namespace anofox {

namespace {

void ValidateDevice(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_device cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "auto" && value != "cpu" && value != "cuda" && value != "rocm" && value != "migraphx" &&
	    value != "coreml") {
		throw InvalidInputException("anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm', 'coreml' "
		                            "('migraphx' is accepted as an alias for 'rocm'), got '%s'",
		                            value);
	}
	parameter = Value(value == "migraphx" ? "rocm" : value);
}

void ValidateTraceLevel(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_trace_level cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "error" && value != "warn" && value != "info" && value != "debug" && value != "trace") {
		throw InvalidInputException(
		    "anofox_tabfm_trace_level must be one of 'error', 'warn', 'info', 'debug', 'trace', got '%s'", value);
	}
	parameter = Value(value);
}

void ValidateGpuPrecision(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_gpu_precision cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "fp32" && value != "bf16" && value != "fp16") {
		throw InvalidInputException("anofox_tabfm_gpu_precision must be 'bf16', 'fp16' or 'fp32', got '%s'", value);
	}
	parameter = Value(value);
}

void ValidatePositive(const char *name, ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("%s cannot be NULL", name);
	}
	auto value = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (value <= 0) {
		throw InvalidInputException("%s must be positive, got %lld", name, value);
	}
}

void ValidateThreads(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_threads", context, scope, parameter);
}

void ValidateMaxRows(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_max_rows", context, scope, parameter);
}

void ValidateMaxFeatures(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_max_features", context, scope, parameter);
}

} // anonymous namespace

void RegisterTabfmSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	config.AddExtensionOption("anofox_tabfm_accept_hf_license",
	                          "Accept the upstream model license (tabfm-non-commercial-v1.0: non-commercial use, "
	                          "no redistribution). Downloads of Google-licensed weights fail without this.",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));

	config.AddExtensionOption("anofox_tabfm_cache_dir",
	                          "Weight cache root directory (default ~/.cache/anofox-tabfm)", LogicalType::VARCHAR,
	                          Value("~/.cache/anofox-tabfm"));

	// Half the cores, but never more than kMaxDefaultThreads: the model's useful intra-op
	// parallelism is a property of the graph, not of the machine, and measuring it on real
	// tabicl-v2 weights (105 MB) puts it at 4-8 threads on both workload shapes. Sweeping 1..12
	// on a 12-core box -- so never oversubscribed -- wall-clock against CPU burned:
	//
	//   500 features x 100 rows    1:2.34s/4.4s   4:1.41s/5.5s   8:1.39s/8.8s  12:1.44s/11.5s
	//   3000 rows x 8 features     1:1.87s/3.6s   4:0.96s/4.0s   8:0.91s/6.0s  12:0.92s/7.2s
	//
	// Past 8 nothing gets faster and CPU keeps climbing. Forcing the counts a 64-core host would
	// pick: 32 threads costs 1.57s/12.2s on the wide shape against 8 threads' 1.40s/7.5s -- 12%
	// slower for 63% more CPU. That surplus is not idle, it is contending, which is what turns a
	// pod with several concurrent sessions into a load average of 153 against 64 cores.
	//
	// Only the default is capped; anofox_tabfm_threads remains settable for anyone whose model or
	// batch scales further, and a host with 16 or fewer cores is unaffected.
	static constexpr int64_t kMaxDefaultThreads = 8;
	const auto half_cores = MaxValue<int64_t>(1, static_cast<int64_t>(std::thread::hardware_concurrency()) / 2);
	const auto default_threads = MinValue<int64_t>(half_cores, kMaxDefaultThreads);
	config.AddExtensionOption("anofox_tabfm_threads", "ONNX Runtime intra-op thread count for CPU inference",
	                          LogicalType::BIGINT, Value::BIGINT(default_threads), ValidateThreads);

	config.AddExtensionOption("anofox_tabfm_max_rows", "Maximum rows per predict call or group",
	                          LogicalType::BIGINT, Value::BIGINT(10000), ValidateMaxRows);

	config.AddExtensionOption("anofox_tabfm_max_features", "Maximum feature columns per predict call",
	                          LogicalType::BIGINT, Value::BIGINT(500), ValidateMaxFeatures);

	config.AddExtensionOption("anofox_tabfm_default_model",
	                          "Default model id for tabfm_classify/regress/download/... when model := is not given. "
	                          "'' = resolve to the single-file manifest model, else the sole registered model.",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption("anofox_tabfm_trace_level", "Diagnostic verbosity: error|warn|info|debug|trace",
	                          LogicalType::VARCHAR, Value("warn"), ValidateTraceLevel);

	config.AddExtensionOption(
	    "anofox_tabfm_gpu_precision",
	    "MIGraphX compile precision on the ROCm GPU: bf16|fp16|fp32. bf16 (default) runs ~2x faster than fp32 on "
	    "RDNA4 and halves VRAM/.mxr, keeping fp32's exponent range; fp32 is the accuracy reference.",
	    LogicalType::VARCHAR, Value("bf16"), ValidateGpuPrecision);

	config.AddExtensionOption(
	    "anofox_tabfm_cpu_prepack",
	    "Enable ONNX Runtime weight prepacking on the CPU EP: faster matmuls at ~+16% resident memory.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption("anofox_tabfm_device",
	                          "Execution device: auto|cpu|cuda|rocm|coreml ('migraphx' alias). Each flavor errors "
	                          "helpfully on devices it does not carry.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory with ONNX Runtime provider / plugin-EP shared libraries",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption(
	    "anofox_tabfm_mxr_source",
	    "Directory holding precompiled MIGraphX .mxr programs (offline/CI/shared cache). Before compiling a "
	    "shape-bucket (~27 min on ROCm), a matching '<model>_<arch>_<precision>_T<t>_H<h>.mxr' here is staged into the "
	    "cache and reused; empty ('' default) always compiles on-device. Artifacts are arch- and ROCm-version-specific.",
	    LogicalType::VARCHAR, Value(""));
}

} // namespace anofox
} // namespace duckdb
