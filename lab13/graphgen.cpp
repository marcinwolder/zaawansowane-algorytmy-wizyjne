// graphgen.cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <stdexcept>

namespace py = pybind11;

// NOTE ON SEMANTICS:
// - Events are processed IN ORDER (sequential), because neighbour_matrix must reflect
//   the latest node index per (x,y) when we add a node.
// - We only parallelize tiny inner loops (if desired) but keep overall logic sequential.
// - Big speedups come from: raw pointer tensor access, single allocation/convert at the end,
//   vector::reserve(), and sparse reset of neighbour_matrix.

class GraphGen {
public:
    GraphGen(int r, int dimension_XY = 32, bool self_loop = true)
        : r_(r), dim_(dimension_XY), self_loop_(self_loop) {

        if (r_ <= 0) throw std::invalid_argument("r must be > 0");
        if (dim_ <= 0) throw std::invalid_argument("dimension_XY must be > 0");

        neighbour_matrix_ = torch::full({dim_, dim_}, -1, torch::dtype(torch::kInt32));
        precompute_offsets();

        // Reserve some default capacity to reduce reallocations
        pos_.reserve(2048);
        feat_.reserve(2048);
        edges_.reserve(16384);
        used_coords_.reserve(2048);
    }

    void reset() {
        // Sparse reset: only clear cells we actually wrote
        int* nmat = neighbour_matrix_.data_ptr<int>();
        for (auto &xy : used_coords_) {
            const int x = xy.first;
            const int y = xy.second;
            nmat[x * dim_ + y] = -1;
        }
        used_coords_.clear();

        pos_.clear();
        feat_.clear();
        edges_.clear();
        index_ = 0;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor events) {
        // events: [N, 4] float32 -> (x, y, t, feature). x,y,t may be non-integers but we use int(x),int(y)
        TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
        TORCH_CHECK(events.is_contiguous(), "events must be contiguous");
        TORCH_CHECK(events.dtype() == torch::kFloat32, "events must be float32");
        TORCH_CHECK(!events.is_cuda(), "CPU version expects events on CPU");

        const int64_t N = events.size(0);
        const float* ev = events.data_ptr<float>();

        // Heuristic reserve based on N
        pos_.reserve(pos_.size() + N);
        feat_.reserve(feat_.size() + N);
        edges_.reserve(edges_.size() + (size_t)N * 8);

        int* nmat = neighbour_matrix_.data_ptr<int>(); // raw pointer access

        for (int64_t i = 0; i < N; ++i) {
            const float xf = ev[i * 4 + 0];
            const float yf = ev[i * 4 + 1];
            const float tf = ev[i * 4 + 2];
            const float feature = ev[i * 4 + 3];

            // Clamp / bounds check
            const int x = (int)xf;
            const int y = (int)yf;
            if ((unsigned)x >= (unsigned)dim_ || (unsigned)y >= (unsigned)dim_) {
                continue;
            }

            // Duplicate check: if there's already a node at (x,y) with same t, skip
            int& cell = nmat[x * dim_ + y];
            if (cell != -1) {
                const auto &prev = pos_[(size_t)cell];
                // prev[2] holds t (float)
                if (prev[2] == tf) {
                    continue;
                }
            }

            // Add node
            pos_.push_back({(float)x, (float)y, tf});
            feat_.push_back(feature);
            const int cur_index = index_;

            if (self_loop_) {
                edges_.push_back({cur_index, cur_index});
            }

            // For each spatial neighbor, if present, check spatio-temporal distance
            // ctx_offsets_ size is small (~pi r^2), so this loop is cheap.
            for (const auto &off : ctx_offsets_) {
                int nx = x + off.first;
                int ny = y + off.second;

                // Clamp to [0, dim_-1]
                if (nx < 0) nx = 0; else if (nx >= dim_) nx = dim_ - 1;
                if (ny < 0) ny = 0; else if (ny >= dim_) ny = dim_ - 1;

                const int nidx = nmat[nx * dim_ + ny];
                if (nidx == -1) continue;

                const auto &npos = pos_[(size_t)nidx];
                const float dx = npos[0] - (float)x;
                const float dy = npos[1] - (float)y;
                const float dt = npos[2] - tf;

                const float dist2 = dx*dx + dy*dy + dt*dt;
                if (dist2 <= r2f_) {
                    edges_.push_back({cur_index, nidx});
                }
            }

            // Update occupancy
            if (cell == -1) {
                used_coords_.push_back({x, y});
            }
            cell = cur_index;
            index_++;
        }

        // Convert to tensors once (no per-event cat)
        const int64_t num_nodes = (int64_t)pos_.size();
        const int64_t num_edges = (int64_t)edges_.size();

        torch::Tensor pos_tensor = torch::empty({num_nodes, 3}, torch::kFloat32);
        torch::Tensor feat_tensor = torch::empty({num_nodes, 1}, torch::kFloat32);
        torch::Tensor edge_tensor = torch::empty({num_edges, 2}, torch::kInt32);

        float* ppos = pos_tensor.data_ptr<float>();
        float* pfeat = feat_tensor.data_ptr<float>();
        int* pedge = edge_tensor.data_ptr<int>();

        // Copy out
        for (int64_t i = 0; i < num_nodes; ++i) {
            const auto &v = pos_[(size_t)i];
            ppos[i * 3 + 0] = v[0];
            ppos[i * 3 + 1] = v[1];
            ppos[i * 3 + 2] = v[2];
            pfeat[i * 1 + 0] = feat_[(size_t)i];
        }
        for (int64_t i = 0; i < num_edges; ++i) {
            const auto &e = edges_[(size_t)i];
            pedge[i * 2 + 0] = e[0];
            pedge[i * 2 + 1] = e[1];
        }

        // IMPORTANT: we reset internal state for next sample but keep memory capacity
        reset();

        return {feat_tensor, pos_tensor, edge_tensor};
    }

private:
    int r_;
    int dim_;
    bool self_loop_;
    int index_ = 0;

    float r2f_ = 0.f; // r^2 in float
    torch::Tensor neighbour_matrix_; // int32 [dim, dim]

    std::vector<std::array<float, 3>> pos_;
    std::vector<float> feat_;
    std::vector<std::array<int, 2>> edges_;
    std::vector<std::pair<int,int>> used_coords_;
    std::vector<std::pair<int,int>> ctx_offsets_;

    void precompute_offsets() {
        r2f_ = float(r_) * float(r_);
        ctx_offsets_.clear();
        ctx_offsets_.reserve((size_t)((2*r_+1)*(2*r_+1)));

        for (int dx = -r_; dx <= r_; ++dx) {
            for (int dy = -r_; dy <= r_; ++dy) {
                if (dx*dx + dy*dy <= r_ * r_) {
                    ctx_offsets_.push_back({dx, dy});
                }
            }
        }
        // Optional: stable order (already stable by loops)
    }
};

// Pybind11 module
PYBIND11_MODULE(graphgen, m) {
    m.doc() = "Fast CPU GraphGen (sequential by time) for event-based graphs";
    py::class_<GraphGen>(m, "GraphGen")
        .def(py::init<int,int,bool>(), py::arg("r"), py::arg("dimension_XY") = 32, py::arg("self_loop") = true)
        .def("forward", &GraphGen::forward, py::arg("events"),
             "events: torch.float32 [N,4] -> returns (x: [Nnodes,1], pos:[Nnodes,3], edge_index:[Nedges,2])")
        .def("reset", &GraphGen::reset);
}
