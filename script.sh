NUM=2
python eval.py -n $NUM -i traces/trace_lam_1.0.txt -o traces/reranking_output_1.0.txt
python eval.py -n $NUM -i traces/trace_lam_0.9.txt -o traces/reranking_output_0.9.txt
python eval.py -n $NUM -i traces/trace_lam_0.8.txt -o traces/reranking_output_0.8.txt
python eval.py -n $NUM -i traces/trace_lam_0.7.txt -o traces/reranking_output_0.7.txt
python eval.py -n $NUM -i traces/trace_lam_0.6.txt -o traces/reranking_output_0.6.txt
python eval.py -n $NUM -i traces/trace_lam_0.5.txt -o traces/reranking_output_0.5.txt
python eval.py -n $NUM -i traces/trace_lam_0.4.txt -o traces/reranking_output_0.4.txt
python eval.py -n $NUM -i traces/trace_lam_0.3.txt -o traces/reranking_output_0.3.txt
python eval.py -n $NUM -i traces/trace_lam_0.2.txt -o traces/reranking_output_0.2.txt
python eval.py -n $NUM -i traces/trace_lam_0_1.txt -o traces/reranking_output_0.1.txt
