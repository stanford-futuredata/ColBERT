#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <utility>

void prune_centroid_scores(const torch::Tensor& centroid_scores,
                           float centroid_score_threshold, bool* idx) {
    auto ncentroids = centroid_scores.size(0);
    auto nquery_vectors = centroid_scores.size(1);
    auto centroid_scores_a = centroid_scores.data_ptr<float>();

    for (int i = 0; i < ncentroids; i++) {
        float max_centroid_score =
            *(std::max_element(centroid_scores_a + (i * nquery_vectors),
                               centroid_scores_a + ((i + 1) * nquery_vectors)));
        idx[i] = max_centroid_score >= centroid_score_threshold;
    }
}

torch::Tensor filter_pids(const torch::Tensor pids,
                          const torch::Tensor centroid_scores,
                          const torch::Tensor codes,
                          const torch::Tensor doclens,
                          const torch::Tensor offsets, const torch::Tensor idx,
                          int nfiltered_docs) {
    auto ncentroids = centroid_scores.size(0);
    auto nquery_vectors = centroid_scores.size(1);
    auto npids = pids.size(0);

    auto pids_a = pids.data_ptr<int>();
    auto centroid_scores_a = centroid_scores.data_ptr<float>();
    auto codes_a = codes.data_ptr<int>();
    auto doclens_a = doclens.data_ptr<int64_t>();
    auto offsets_a = offsets.data_ptr<int64_t>();
    auto idx_a = idx.data_ptr<bool>();

    float per_doc_approx_scores[nquery_vectors];
    for (int k = 0; k < nquery_vectors; k++) {
        per_doc_approx_scores[k] = -9999;
    }
    std::priority_queue<std::pair<float, int>> approx_scores;
    for (int i = 0; i < npids; i++) {
        std::set<int> seen_codes;
        for (int j = 0; j < doclens_a[pids_a[i]]; j++) {
            auto code = codes_a[offsets_a[pids_a[i]] + j];
            assert(code < ncentroids);
            if (idx_a[code] && seen_codes.find(code) == seen_codes.end()) {
                std::transform(per_doc_approx_scores,
                               per_doc_approx_scores + nquery_vectors,
                               centroid_scores_a + (code * nquery_vectors),
                               per_doc_approx_scores,
                               [](float a, float b) { return std::max(a, b); });
                seen_codes.insert(code);
            }
        }
        float score = 0;
        for (int k = 0; k < nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }
        approx_scores.push(std::make_pair(score, pids_a[i]));
    }

    int filtered_pids[nfiltered_docs];
    for (int i = 0; i < nfiltered_docs; i++) {
        std::pair<float, int> score_and_pid = approx_scores.top();
        filtered_pids[i] = score_and_pid.second;
        approx_scores.pop();
    }

    approx_scores = std::priority_queue<std::pair<float, int>>();
    for (int i = 0; i < nfiltered_docs; i++) {
        int pid = filtered_pids[i];
        for (int j = 0; j < doclens_a[pid]; j++) {
            auto code = codes_a[offsets_a[pid] + j];
            assert(code < ncentroids);
            std::transform(per_doc_approx_scores,
                           per_doc_approx_scores + nquery_vectors,
                           centroid_scores_a + (code * nquery_vectors),
                           per_doc_approx_scores,
                           [](float a, float b) { return std::max(a, b); });
        }
        float score = 0;
        for (int k = 0; k < nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }
        approx_scores.push(std::make_pair(score, pid));
    }

    int nfinal_filtered_docs = (int)(nfiltered_docs / 4);
    int final_filtered_pids[nfinal_filtered_docs];
    for (int i = 0; i < nfinal_filtered_docs; i++) {
        std::pair<float, int> score_and_pid = approx_scores.top();
        final_filtered_pids[i] = score_and_pid.second;
        approx_scores.pop();
    }

    auto options =
        torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    return torch::from_blob(final_filtered_pids, {nfinal_filtered_docs},
                            options)
        .clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_pids_cpp", &filter_pids, "Filter pids");
}

