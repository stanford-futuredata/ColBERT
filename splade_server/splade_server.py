import grpc
import torch
import splade_pb2
import splade_pb2_grpc
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from concurrent import futures

class SpladeServer(splade_pb2_grpc.SpladeServicer):
    def __init__(self):
        model_type_or_dir = "naver/efficient-splade-V-large-query"
        self.model = Splade(model_type_or_dir, agg="max")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)


    def gen_query(self, query_text, multiplier):
        with torch.no_grad():
            query_rep = torch.trunc(multiplier * self.model(q_kwargs=self.tokenizer(query_text, return_tensors="pt"))["q_rep"].squeeze()).int()
        col = torch.nonzero(query_rep).squeeze().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(col)
        counts = query_rep[col].cpu().tolist()
        query = ' '.join(word for word, count in zip(tokens, counts) for _ in range(count))
        return splade_pb2.QueryStr(query = query, multiplier = multiplier)

    def GenerateQuery(self, request, context):
        return self.gen_query(request.query, request.multiplier)

def serve_Splade_server():
    server = grpc.server(futures.ThreadPoolExecutor())
    splade_pb2_grpc.add_SpladeServicer_to_server(SpladeServer(), server)
    listen_addr = '[::]:50060'
    server.add_insecure_port(listen_addr)
    print(f"Starting Splade server on {listen_addr}")
    server.start()
    server.wait_for_termination()
    print("Terminated")


if __name__ == '__main__':
    serve_Splade_server()
