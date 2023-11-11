import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class CodeT5pEncDecReranker(nn.Module):

    def __init__(self, checkpoint):
        super().__init__()

        self.codet5p = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.GOOD_TOKEN = self.raw_tokenizer.encode('good')[0]

    def forward(self, encoding, decoder_input_ids=None):
        # If decoder_input_ids are not provided, use input_ids shifted by one token
        if decoder_input_ids is None:
            decoder_input_ids = encoding.input_ids[:, 1:2]

        # Get logits from the model
        outputs = self.codet5p(input_ids=encoding.input_ids,
                            attention_mask=encoding.attention_mask,
                            decoder_input_ids=decoder_input_ids)

        logits = outputs.logits
        scores = logits[:, -1, self.GOOD_TOKEN]  # Assuming you want the score of the last token position

        return scores

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.codet5p.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)
