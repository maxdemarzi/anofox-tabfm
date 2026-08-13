#include "catch.hpp"

#include "tabfm_ckpt.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {
std::vector<uint8_t> ReadFile(const char *path) {
	FILE *f = fopen(path, "rb");
	REQUIRE(f != nullptr);
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<uint8_t> buf(n);
	REQUIRE(fread(buf.data(), 1, n, f) == (size_t)n);
	fclose(f);
	return buf;
}
} // namespace

TEST_CASE("tabfm_ckpt: native torch .ckpt reader recovers the state_dict", "[tabfm][ckpt]") {
	auto buf = ReadFile("test/fixtures/ckpt/tiny.ckpt");
	REQUIRE(IsTorchCkpt(buf.data(), buf.size()));

	auto sd = ReadTorchCkpt(buf.data(), buf.size());
	REQUIRE(sd.size() == 3);
	REQUIRE(sd.count("layer.weight"));
	REQUIRE(sd.count("layer.bias"));
	REQUIRE(sd.count("idx"));

	// f32 [2,3] = 0..5
	auto &w = sd.at("layer.weight");
	REQUIRE(w.dtype == "f32");
	REQUIRE(w.shape == vector<int64_t> {2, 3});
	REQUIRE(w.nbytes == 6 * sizeof(float));
	const float *wf = reinterpret_cast<const float *>(w.data);
	for (int i = 0; i < 6; i++) {
		REQUIRE(wf[i] == Approx((float)i));
	}

	// f32 [2] = {1.5, -2.5}
	auto &b = sd.at("layer.bias");
	REQUIRE(b.dtype == "f32");
	REQUIRE(b.shape == vector<int64_t> {2});
	const float *bf = reinterpret_cast<const float *>(b.data);
	REQUIRE(bf[0] == Approx(1.5f));
	REQUIRE(bf[1] == Approx(-2.5f));

	// i64 [1,3] = {7,8,9}
	auto &idx = sd.at("idx");
	REQUIRE(idx.dtype == "i64");
	REQUIRE(idx.shape == vector<int64_t> {1, 3});
	const int64_t *ii = reinterpret_cast<const int64_t *>(idx.data);
	REQUIRE(ii[0] == 7);
	REQUIRE(ii[1] == 8);
	REQUIRE(ii[2] == 9);
}

TEST_CASE("tabfm_ckpt: non-zip buffer is rejected", "[tabfm][ckpt]") {
	const char *garbage = "not a zip file at all";
	REQUIRE_FALSE(IsTorchCkpt(reinterpret_cast<const uint8_t *>(garbage), strlen(garbage)));
}

//===--------------------------------------------------------------------===//
// Hostile archives.
//
// A .ckpt is attacker-controllable input: tabfm_download fetches one from a
// HuggingFace repo and ReadZip parses it before anything has been validated. The
// central directory's length fields are 16- and 32-bit values straight out of
// the file, so each one needs bounding against the buffer we actually hold.
//===--------------------------------------------------------------------===//

namespace {

void PutU16(std::vector<uint8_t> &out, uint16_t v) {
	out.push_back(uint8_t(v));
	out.push_back(uint8_t(v >> 8));
}
void PutU32(std::vector<uint8_t> &out, uint32_t v) {
	for (int i = 0; i < 4; i++) {
		out.push_back(uint8_t(v >> (8 * i)));
	}
}

//! A minimal single-entry archive. `fn_len` and `lho` are written verbatim so a
//! test can declare values the file does not back.
std::vector<uint8_t> BuildZip(uint16_t fn_len, uint32_t lho, uint16_t method) {
	std::vector<uint8_t> out;
	PutU32(out, 0x04034b50); // local file header, so IsTorchCkpt matches
	PutU16(out, 20);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU32(out, 0);
	PutU32(out, 0);
	PutU32(out, 0);
	PutU16(out, 1);
	PutU16(out, 0);
	out.push_back('x');

	const uint32_t cd_offset = uint32_t(out.size());
	PutU32(out, 0x02014b50); // central directory header
	PutU16(out, 20);
	PutU16(out, 20);
	PutU16(out, 0);
	PutU16(out, method);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU32(out, 0);
	PutU32(out, 0);
	PutU32(out, 0);
	PutU16(out, fn_len); // declared filename length
	PutU16(out, 0);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU16(out, 0);
	PutU32(out, 0);
	PutU32(out, lho); // declared local header offset
	out.push_back('x'); // ...but only one byte of filename is present
	const uint32_t cd_size = uint32_t(out.size()) - cd_offset;

	PutU32(out, 0x06054b50); // end of central directory
	PutU16(out, 0);
	PutU16(out, 0);
	PutU16(out, 1);
	PutU16(out, 1);
	PutU32(out, cd_size);
	PutU32(out, cd_offset);
	PutU16(out, 0);
	return out;
}

} // namespace

TEST_CASE("tabfm_ckpt: a filename length past the end of the buffer is rejected", "[tabfm][ckpt]") {
	// fn_len claims 60000 bytes of filename in a ~100 byte file. Unbounded, the
	// name is built from 60000 bytes of whatever follows the buffer -- and it is
	// interpolated into the error message, so the read is also disclosed.
	//
	// method=0 (STORED) on purpose: a compressed entry throws for an unrelated
	// reason and would mask this. The assertion is on *why* the file is rejected,
	// because "it threw something" was already true while the read was happening.
	auto buf = BuildZip(/*fn_len=*/60000, /*lho=*/0, /*method=*/0);
	REQUIRE(IsTorchCkpt(buf.data(), buf.size()));
	std::string message;
	try {
		ReadTorchCkpt(buf.data(), buf.size());
		FAIL("expected a malformed-archive error");
	} catch (const InvalidInputException &e) {
		message = e.what();
	}
	REQUIRE(message.find("central directory") != std::string::npos);
}

TEST_CASE("tabfm_ckpt: a local header offset near 2^32 does not wrap the bounds check",
          "[tabfm][ckpt]") {
	// `lho + 30` computed in 32-bit arithmetic wraps to 29 for this offset, passes
	// a `> size` test, and then reads ~4 GB out of bounds.
	auto buf = BuildZip(/*fn_len=*/1, /*lho=*/0xFFFFFFFFu, /*method=*/0);
	REQUIRE(IsTorchCkpt(buf.data(), buf.size()));
	REQUIRE_THROWS_AS(ReadTorchCkpt(buf.data(), buf.size()), InvalidInputException);
}
