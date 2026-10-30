// Which CUDA device each logical card sits on. Decode's sums and the attention merge write only to the next card
// (kernels.cu's ring, n - 1 hops to reduce and n - 1 to spread, 96 sums a window on this model), so a hop that leaves
// NVLink is on the critical path of every one of them. An 8-GPU V100 meshes NVLink over some pairs and PCIe over the
// rest; PCI order walks the slow pairs. The order is a permutation of the visible devices. Unset keeps PCI order,
// which is what two, three and four cards were measured with.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fnx {

// cudaDeviceGetP2PAttribute failed for that directed pair: treated as slower than any real rank
constexpr int RING_RANK_NONE = 1000000;

enum class RingOrderKind { Pci, Auto, Explicit };

struct RingOrderRequest {
    RingOrderKind kind = RingOrderKind::Pci;
    std::vector<int> phys;   // Explicit: cuda device of logical card 0, 1, ...
};

// FNX_RING_ORDER: unset, "off" or "pci" keeps PCI order; "auto" asks ring_order_auto; "0,1,3,..." is that permutation
// of the visible devices. Throws on anything else, naming the value.
inline RingOrderRequest parse_ring_order(const char* env, int n) {
    if (n < 2) throw std::invalid_argument("parse_ring_order: " + std::to_string(n) + " cards");
    RingOrderRequest r;
    if (!env || !env[0]) return r;
    std::string s(env);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t')) ++a;
    s = s.substr(a);
    if (s == "off" || s == "pci") return r;
    if (s == "auto") {
        r.kind = RingOrderKind::Auto;
        return r;
    }
    std::vector<int> phys;
    std::string cur;
    const auto push = [&] {
        while (!cur.empty() && (cur.back() == ' ' || cur.back() == '\t')) cur.pop_back();
        size_t b = 0;
        while (b < cur.size() && (cur[b] == ' ' || cur[b] == '\t')) ++b;
        cur = cur.substr(b);
        if (cur.empty()) throw std::runtime_error(std::string("FNX_RING_ORDER: '") + env + "' has an empty device");
        char* end = nullptr;
        const long v = std::strtol(cur.c_str(), &end, 10);
        if (end == cur.c_str() || *end) throw std::runtime_error(std::string("FNX_RING_ORDER: '") + env + "' is not off, pci, auto, or cuda devices");
        phys.push_back((int) v);
        cur.clear();
    };
    for (char ch : s) {
        if (ch == ',') push();
        else cur.push_back(ch);
    }
    push();
    if ((int) phys.size() != n) throw std::runtime_error("FNX_RING_ORDER: '" + s + "' has " + std::to_string(phys.size()) + " devices, want " + std::to_string(n));
    std::vector<char> seen((size_t) n, 0);
    for (int d : phys) {
        if (d < 0 || d >= n || seen[(size_t) d])
            throw std::runtime_error("FNX_RING_ORDER: '" + s + "' is not a permutation of cuda devices 0.." + std::to_string(n - 1));
        seen[(size_t) d] = 1;
    }
    r.kind = RingOrderKind::Explicit;
    r.phys = std::move(phys);
    return r;
}

// Directed P2P ranks, lower faster (cudaDevP2PAttrPerformanceRank): ranks[i * n + j] is device i writing device j.
// A cycle on every device, rotated so device 0 is logical card 0. Minimises the worst hop, then the sum of hop ranks.
// PCI order wins a tie, so a machine whose links are all the same rank does not move. n is at most 8: the search is
// the (n - 1)! orders that start at device 0.
inline std::vector<int> ring_order_auto(int n, const std::vector<int>& ranks) {
    if (n < 2 || (int) ranks.size() != n * n)
        throw std::invalid_argument("ring_order_auto: " + std::to_string(ranks.size()) + " ranks for " + std::to_string(n) + " cards");
    struct Search {
        int n;
        const std::vector<int>& ranks;
        std::pair<int, int> best;
        std::vector<int> win, perm;
        std::vector<char> used;
        int edge(int i, int j) const { return ranks[(size_t) i * n + j]; }
        std::pair<int, int> score(const std::vector<int>& p) const {
            int worst = 0, sum = 0;
            for (int i = 0; i < n; ++i) {
                const int r = edge(p[(size_t) i], p[(size_t) (i + 1) % n]);
                if (r > worst) worst = r;
                sum += r;
            }
            return {worst, sum};
        }
        void rec(int k) {
            if (k == n) {
                const std::pair<int, int> s = score(perm);
                if (s < best) {   // search is lexicographic, so the first optimum is the smallest order
                    best = s;
                    win = perm;
                }
                return;
            }
            for (int v = 1; v < n; ++v) {
                if (used[(size_t) v]) continue;
                if (edge(perm[(size_t) k - 1], v) > best.first) continue;   // this hop alone is worse than a cycle already found
                used[(size_t) v] = 1;
                perm[(size_t) k] = v;
                rec(k + 1);
                used[(size_t) v] = 0;
            }
        }
    };
    Search s{n, ranks, {}, std::vector<int>(n), std::vector<int>(n), std::vector<char>((size_t) n, 0)};
    for (int i = 0; i < n; ++i) s.win[(size_t) i] = i;
    s.best = s.score(s.win);
    s.perm[0] = 0;
    s.used[0] = 1;
    s.rec(1);
    return s.win;
}

// one line for stderr. ranks empty, or n * n directed ranks as ring_order_auto takes them
inline std::string ring_order_message(RingOrderKind kind, const std::vector<int>& phys, const std::vector<int>& ranks) {
    const int n = (int) phys.size();
    if (n < 2) throw std::invalid_argument("ring_order_message: " + std::to_string(n) + " cards");
    std::string devs;
    for (int i = 0; i < n; ++i) devs += (i ? " " : "") + std::to_string(phys[(size_t) i]);
    const char* why = "PCI order";
    if (kind == RingOrderKind::Auto) why = "FNX_RING_ORDER=auto";
    else if (kind == RingOrderKind::Explicit) why = "FNX_RING_ORDER";
    std::string hops;
    if ((int) ranks.size() == n * n) {
        int known = 0, best = RING_RANK_NONE;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const int r = ranks[(size_t) i * n + j];
                if (r == RING_RANK_NONE) continue;
                ++known;
                if (r < best) best = r;
            }
        if (known == 0) {
            hops = "; P2P ranks unavailable";
            if (kind == RingOrderKind::Auto) hops += ", so this is PCI order";   // every rank missing leaves the search on PCI order
        }
        else {
            hops = "; hop ranks";
            int worst = 0;
            for (int i = 0; i < n; ++i) {
                const int r = ranks[(size_t) phys[(size_t) i] * n + phys[(size_t) (i + 1) % n]];
                hops += " " + std::to_string(r);
                if (r > worst) worst = r;
            }
            hops += " (best rank in the machine " + std::to_string(best) + ", worst hop " + std::to_string(worst) + ")";
        }
    }
    return std::string("fnx: ring ") + why + ": logical cards on cuda devices " + devs + hops;
}

// the holder placed the weights (holder[logical] = cuda device). A worker whose FNX_RING_ORDER would place them
// elsewhere would read the right slice off the wrong card
inline std::string ring_order_mismatch(const std::vector<int>& holder, const std::vector<int>& want) {
    auto list = [](const std::vector<int>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
        return s;
    };
    return "FNX_RING_ORDER puts logical cards on cuda devices " + list(want) + " but the holder placed them on " + list(holder) +
           ": unload the lab and start it again with the same FNX_RING_ORDER";
}

// gbs[from * n + to] is that directed pair's kernel-store GB/s, negative when the probe has no rate. The ring in
// logical order is card c -> c + 1. A hop under half the fastest pair is said: both can be several GB/s, so the
// probe's 1 GB/s line does not catch a PCIe hop beside NVLink.
inline std::vector<std::string> ring_hop_warnings(const std::vector<double>& gbs, int n) {
    if (n < 2 || (int) gbs.size() != n * n)
        throw std::invalid_argument("ring_hop_warnings: " + std::to_string(gbs.size()) + " rates for " + std::to_string(n) + " cards");
    double mx = 0;
    for (double g : gbs)
        if (g > mx) mx = g;
    std::vector<std::string> out;
    if (!(mx > 0)) return out;
    for (int c = 0; c < n; ++c) {
        const int d = (c + 1) % n;
        const double g = gbs[(size_t) c * n + (size_t) d];
        if (!(g >= 0) || !(g < mx * 0.5)) continue;
        char line[512];
        const int nw = std::snprintf(line, sizeof line,
                                     "fnx: warning: ring hop logical card %d -> %d stores at %.1f GB/s, under half the fastest pair "
                                     "(%.1f GB/s). Decode's sums cross only the next card, so this hop is on every one of them. "
                                     "FNX_RING_ORDER=auto picks the ring from P2P ranks; or set FNX_RING_ORDER to cuda devices in ring order",
                                     c, d, g, mx);
        if (nw < 0 || nw >= (int) sizeof line) throw std::logic_error("ring_hop_warnings: a line past the buffer");
        out.push_back(line);
    }
    return out;
}

}  // namespace fnx
