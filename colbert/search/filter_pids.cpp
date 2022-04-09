#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <utility>

typedef struct maxsim_args {
    int tid;
    int nthreads;

    int ncentroids;
    int nquery_vectors;
    int npids;

    int* pids;
    float* centroid_scores;
    int* codes;
    int64_t* doclens;
    int64_t* offsets;
    bool* idx;

    std::priority_queue<std::pair<float, int>> approx_scores;
} maxsim_args_t;

void* maxsim(void* args) {
    maxsim_args_t* maxsim_args = (maxsim_args_t*)args;

    float per_doc_approx_scores[maxsim_args->nquery_vectors];
    for (int k = 0; k < maxsim_args->nquery_vectors; k++) {
        per_doc_approx_scores[k] = -9999;
    }

    int ndocs_per_thread =
        (int)std::ceil(((float)maxsim_args->npids) / maxsim_args->nthreads);
    int start = maxsim_args->tid * ndocs_per_thread;
    int end =
        std::min((maxsim_args->tid + 1) * ndocs_per_thread, maxsim_args->npids);

    for (int i = start; i < end; i++) {
        std::set<int> seen_codes;
        auto pid = maxsim_args->pids[i];
        for (int j = 0; j < maxsim_args->doclens[pid]; j++) {
            auto code = maxsim_args->codes[maxsim_args->offsets[pid] + j];
            assert(code < maxsim_args->ncentroids);
            if (maxsim_args->idx[code] &&
                seen_codes.find(code) == seen_codes.end()) {
                std::transform(
                    per_doc_approx_scores,
                    per_doc_approx_scores + maxsim_args->nquery_vectors,
                    maxsim_args->centroid_scores +
                        (code * maxsim_args->nquery_vectors),
                    per_doc_approx_scores,
                    [](float a, float b) { return std::max(a, b); });
                seen_codes.insert(code);
            }
        }
        float score = 0;
        for (int k = 0; k < maxsim_args->nquery_vectors; k++) {
            score += per_doc_approx_scores[k];
            per_doc_approx_scores[k] = -9999;
        }
        maxsim_args->approx_scores.push(std::make_pair(score, pid));
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

    int nthreads = at::get_num_threads();

    pthread_t threads[nthreads];
    maxsim_args_t args[nthreads];

    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].nthreads = nthreads;

        args[i].ncentroids = ncentroids;
        args[i].nquery_vectors = nquery_vectors;
        args[i].npids = npids;

        args[i].pids = pids_a;
        args[i].centroid_scores = centroid_scores_a;
        args[i].codes = codes_a;
        args[i].doclens = doclens_a;
        args[i].offsets = offsets_a;
        args[i].idx = idx_a;

        args[i].approx_scores = std::priority_queue<std::pair<float, int>>();

        int rc = pthread_create(&threads[i], NULL, maxsim, (void*)&args[i]);
        if (rc) {
            fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    std::priority_queue<std::pair<float, int>> global_approx_scores;
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < nfiltered_docs; j++) {
            global_approx_scores.push(args[i].approx_scores.top());
            args[i].approx_scores.pop();
        }
    }

    int filtered_pids[nfiltered_docs];
    for (int i = 0; i < nfiltered_docs; i++) {
        std::pair<float, int> score_and_pid = global_approx_scores.top();
        filtered_pids[i] = score_and_pid.second;
        global_approx_scores.pop();
    }

    std::priority_queue<std::pair<float, int>> approx_scores;
    float per_doc_approx_scores[nquery_vectors];
    for (int k = 0; k < nquery_vectors; k++) {
        per_doc_approx_scores[k] = -9999;
    }
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

