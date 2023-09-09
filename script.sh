NUM=2
python eval.py -n $NUM -i traces/trace_lam_1.0.txt -o traces/rerank_output_1.0.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.9.txt -o traces/rerank_output_0.9.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.8.txt -o traces/rerank_output_0.8.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.7.txt -o traces/rerank_output_0.7.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.6.txt -o traces/rerank_output_0.6.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.5.txt -o traces/rerank_output_0.5.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.4.txt -o traces/rerank_output_0.4.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.3.txt -o traces/rerank_output_0.3.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0.2.txt -o traces/rerank_output_0.2.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_0_1.txt -o traces/rerank_output_0.1.txt -e rerank
python eval.py -n $NUM -i traces/trace_lam_1.0.txt -o traces/search_output_1.0.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.9.txt -o traces/search_output_0.9.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.8.txt -o traces/search_output_0.8.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.7.txt -o traces/search_output_0.7.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.6.txt -o traces/search_output_0.6.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.5.txt -o traces/search_output_0.5.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.4.txt -o traces/search_output_0.4.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.3.txt -o traces/search_output_0.3.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0.2.txt -o traces/search_output_0.2.txt -e search
python eval.py -n $NUM -i traces/trace_lam_0_1.txt -o traces/search_output_0.1.txt -e search

