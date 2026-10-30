// The ring's device order (src/ring_order.hpp), on the CPU: PCI order unless FNX_RING_ORDER asks, auto keeps every hop
// on the fastest P2P rank that still makes a cycle, and a hop far slower than the best pair is named. No CUDA.
//   bin/ring_order_test
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../src/ring_order.hpp"

namespace {

int fails = 0;
void fail(const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    ++fails;
}

void eq_perm(const char* name, const std::vector<int>& got, const std::vector<int>& want) {
    if (got != want) {
        std::string a, b;
        for (int v : got) a += std::to_string(v) + " ";
        for (int v : want) b += std::to_string(v) + " ";
        fail(std::string(name) + ": got " + a + "want " + b);
    }
}

std::vector<int> ranks_of(int n, int fill, const std::vector<std::tuple<int, int, int>>& edges) {
    std::vector<int> r((size_t) n * n, fill);
    for (int i = 0; i < n; ++i) r[(size_t) i * n + i] = 0;
    for (const auto& e : edges) r[(size_t) std::get<0>(e) * n + std::get<1>(e)] = std::get<2>(e);
    return r;
}

void kind_is(const char* env, int n, fnx::RingOrderKind want) {
    try {
        const fnx::RingOrderRequest r = fnx::parse_ring_order(env, n);
        if (r.kind != want) fail(std::string("parse '") + (env ? env : "(null)") + "' kind");
    } catch (const std::exception& e) {
        fail(std::string("parse '") + (env ? env : "(null)") + "' threw: " + e.what());
    }
}

void parse_refused(const char* env, int n) {
    try {
        fnx::parse_ring_order(env, n);
        fail(std::string("parse '") + env + "' was accepted");
    } catch (const std::runtime_error&) {
    } catch (const std::exception& e) {
        fail(std::string("parse '") + env + "' threw " + e.what());
    }
}

}  // namespace

int main() {
    kind_is(nullptr, 8, fnx::RingOrderKind::Pci);
    kind_is("", 4, fnx::RingOrderKind::Pci);
    kind_is("off", 8, fnx::RingOrderKind::Pci);
    kind_is("pci", 3, fnx::RingOrderKind::Pci);
    kind_is("  auto  ", 8, fnx::RingOrderKind::Auto);
    parse_refused("Auto", 8);
    parse_refused("0,1,2", 4);
    parse_refused("0,0,1", 3);
    parse_refused("0,,1", 3);
    parse_refused("0,1,8", 3);
    parse_refused("0,1,-1", 3);
    try {
        const fnx::RingOrderRequest r = fnx::parse_ring_order("0, 1, 3, 7, 5, 4, 6, 2", 8);
        if (r.kind != fnx::RingOrderKind::Explicit) fail("explicit kind");
        eq_perm("explicit", r.phys, {0, 1, 3, 7, 5, 4, 6, 2});
    } catch (const std::exception& e) {
        fail(std::string("explicit threw: ") + e.what());
    }

    // every link the same rank: PCI order, including when the rank is "no attribute"
    for (int n : {2, 3, 4, 6, 8}) {
        eq_perm("uniform", fnx::ring_order_auto(n, ranks_of(n, 1, {})), [&] {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[(size_t) i] = i;
            return id;
        }());
        eq_perm("none", fnx::ring_order_auto(n, ranks_of(n, fnx::RING_RANK_NONE, {})), [&] {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[(size_t) i] = i;
            return id;
        }());
    }

    // three cards, the forward hop off card 0 slow and the reverse cycle fast
    {
        const std::vector<int> r = ranks_of(3, 0, {{0, 1, 5}});
        eq_perm("reverse 3", fnx::ring_order_auto(3, r), {0, 2, 1});
    }

    // four cards: identity's first hop is the slow one, every other directed edge is fast. The lexicographically
    // smallest fast cycle that starts at device 0
    {
        const std::vector<int> r = ranks_of(4, 0, {{0, 1, 3}});
        eq_perm("avoid 0->1", fnx::ring_order_auto(4, r), {0, 2, 1, 3});
        for (int i = 0; i < 4; ++i) {
            const std::vector<int> p = fnx::ring_order_auto(4, r);
            const int a = p[(size_t) i], b = p[(size_t) (i + 1) % 4];
            if (r[(size_t) a * 4 + b] != 0) fail("four-card cycle uses the slow hop");
        }
    }

    // DGX-1 V100 hybrid cube, as nvidia-smi topo -m prints it: NV2 (two links) rank 0, NV1 rank 1, PHB rank 2.
    // The double links are themselves an 8-cycle; PCI order uses PHB.
    {
        const int NV2 = 0, NV1 = 1, PHB = 2;
        std::vector<int> r((size_t) 8 * 8, PHB);
        for (int i = 0; i < 8; ++i) r[(size_t) i * 8 + i] = 0;
        const int nv2[][2] = {{0, 3}, {0, 4}, {1, 2}, {1, 5}, {2, 3}, {4, 7}, {5, 6}, {6, 7}};
        const int nv1[][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}};
        for (const auto& e : nv2) {
            r[(size_t) e[0] * 8 + e[1]] = r[(size_t) e[1] * 8 + e[0]] = NV2;
        }
        for (const auto& e : nv1) {
            r[(size_t) e[0] * 8 + e[1]] = r[(size_t) e[1] * 8 + e[0]] = NV1;
        }
        const std::vector<int> got = fnx::ring_order_auto(8, r);
        eq_perm("dgx1", got, {0, 3, 2, 1, 5, 6, 7, 4});
        for (int i = 0; i < 8; ++i) {
            const int a = got[(size_t) i], b = got[(size_t) (i + 1) % 8];
            if (r[(size_t) a * 8 + b] != NV2) fail("dgx1 hop " + std::to_string(a) + " -> " + std::to_string(b) + " is not NV2");
        }
        const std::vector<int> pci = fnx::ring_order_auto(8, ranks_of(8, PHB, {}));
        eq_perm("dgx1 uniform would stay", pci, {0, 1, 2, 3, 4, 5, 6, 7});
        const std::string msg = fnx::ring_order_message(fnx::RingOrderKind::Auto, got, r);
        if (msg != "fnx: ring FNX_RING_ORDER=auto: logical cards on cuda devices 0 3 2 1 5 6 7 4; hop ranks 0 0 0 0 0 0 0 0 "
                   "(best rank in the machine 0, worst hop 0)")
            fail("auto message: " + msg);
    }

    {
        const std::string msg = fnx::ring_order_message(fnx::RingOrderKind::Pci, {0, 1, 2}, {});
        if (msg != "fnx: ring PCI order: logical cards on cuda devices 0 1 2") fail("pci message: " + msg);
        const std::string none = fnx::ring_order_message(fnx::RingOrderKind::Auto, {0, 1, 2, 3}, ranks_of(4, fnx::RING_RANK_NONE, {}));
        if (none.find("P2P ranks unavailable") == std::string::npos) fail("missing ranks: " + none);
        const std::string mis = fnx::ring_order_mismatch({0, 1, 2}, {0, 2, 1});
        if (mis.find("0 2 1") == std::string::npos || mis.find("0 1 2") == std::string::npos || mis.find("unload the lab") == std::string::npos)
            fail("mismatch: " + mis);
    }

    // a PCIe hop beside a much faster pair is named; a uniform machine is not; exactly half the best rate is not
    {
        std::vector<double> g((size_t) 4 * 4, 20);
        for (int i = 0; i < 4; ++i) g[(size_t) i * 4 + i] = -1;
        g[0 * 4 + 1] = 6;   // the PCI-order ring's first hop
        const std::vector<std::string> w = fnx::ring_hop_warnings(g, 4);
        if (w.size() != 1 || w[0].find("logical card 0 -> 1") == std::string::npos || w[0].find("6.0 GB/s") == std::string::npos)
            fail(w.empty() ? "no hop warning" : "hop warning: " + w[0]);
        std::vector<double> even((size_t) 4 * 4, 7);
        if (!fnx::ring_hop_warnings(even, 4).empty()) fail("uniform rates warned");
        std::vector<double> half = even;
        half[0] = 14;   // (0,0) is unused; set a real pair
        half[2 * 4 + 3] = 14;
        half[0 * 4 + 1] = 7;   // exactly half of 14
        if (!fnx::ring_hop_warnings(half, 4).empty()) fail("exactly half warned");
    }

    if (fails) {
        std::printf("%d failed\n", fails);
        return 1;
    }
    std::printf("ring_order_test: ok\n");
    return 0;
}
