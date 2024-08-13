import argparse
import torch

from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer

MARCO_QUERIES = [
    "what color is amber urine",
    "is autoimmune hepatitis a bile acid synthesis disorder",
    "elegxo meaning",
    "how much does an average person make for tutoring",
    "can you use a calculator on the compass test",
    "what does physical medicine do",
    "what does pending mean on listing",
]

MARCO_DOCS = [
    "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.",
    "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.",
    "Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.",
    "The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 â¦ 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves.",
    "versions of each volume as well as complementary websites. The first websiteâThe Manhattan Project: An Interactive Historyâis available on the Office of History and Heritage Resources website, http://www.cfo. doe.gov/me70/history. The Office of History and Heritage Resources and the National Nuclear Security",
    "The Manhattan Project. This once classified photograph features the first atomic bomb â a weapon that atomic scientists had nicknamed Gadget.. The nuclear age began on July 16, 1945, when it was detonated in the New Mexico desert.",
    "Nor will it attempt to substitute for the extraordinarily rich literature on the atomic bombs and the end of World War II. This collection does not attempt to document the origins and development of the Manhattan Project.",
    "Manhattan Project. The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs. The Army component of the project was designated the",
    "In June 1942, the United States Army Corps of Engineersbegan the Manhattan Project- The secret name for the 2 atomic bombs.",
    "One of the main reasons Hanford was selected as a site for the Manhattan Project's B Reactor was its proximity to the Columbia River, the largest river flowing into the Pacific Ocean from the North American coast.",
]


def test_query_basic_tensorize(args: argparse.Namespace) -> None:
    
    print("Testing QueryTokenizer.tensorize shape and marker position")
    config = ColBERTConfig.load_from_checkpoint(args.checkpoint)
    query_tokenizer = QueryTokenizer(config)

    ids, mask = query_tokenizer.tensorize(MARCO_QUERIES)

    if args.verbose:
        print("Tokenized Queries:")
        for example in query_tokenizer.tok.batch_decode(ids):
            print(example, "\n")

        print(f"{ids.shape=}")
    
    assert ids.shape == (len(MARCO_QUERIES), config.query_maxlen), "Ids shape is not as expected"
    assert mask.shape == (len(MARCO_QUERIES), config.query_maxlen), "Mask shape is not as expected"

    # Marker in place
    assert (ids[:, 1] == query_tokenizer.Q_marker_token_id).all(), "Query Marker is not after the first token"

    print("All tests passed!")



def test_doc_basic_tensorize(args: argparse.Namespace) -> None:

    print("Testing DocTokenizer.tensorize shape and marker position")
    config = ColBERTConfig.load_from_checkpoint(args.checkpoint)
    doc_tokenizer = DocTokenizer(config)

    ids, mask = doc_tokenizer.tensorize(MARCO_DOCS)

    if args.verbose:
        print("Tokenized Documents:")
        for example in doc_tokenizer.tok.batch_decode(ids):
            print(example, "\n")

        print(f"{ids.shape=}")

    # Marker in place
    assert (ids[:, 1] == doc_tokenizer.D_marker_token_id).all(), "Document Marker is not after the first token"

    print("All tests passed!")


def test_tensorize_colbert_v2() -> None:
    print("Testing tensorize ids on specific examples for colbertv2.0")
    config = ColBERTConfig.load_from_checkpoint("colbert-ir/colbertv2.0")
    query_tokenizer = QueryTokenizer(config)
    ids, mask = query_tokenizer.tensorize(MARCO_QUERIES[:1])

    if args.verbose:
        print("Tokenized Queries:")
        for example in query_tokenizer.tok.batch_decode(ids):
            print(example, "\n")

    expected_query_ids = torch.tensor([
        101, 1, 2054, 3609, 2003, 8994, 17996, 102, 103, 103, 
        103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 
        103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103,
    ])

    expected_query_mask = torch.tensor([
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    assert (ids == expected_query_ids.to(ids.device)).all(), "Query tokenized ids not as expected for colbertv2.0 on single example"

    assert (mask == expected_query_mask.to(mask.device)).all(), "Query tokenized mask is not as expected for colbertv2.0 on single example"

    doc_tokenizer = DocTokenizer(config)
    ids, mask = doc_tokenizer.tensorize(MARCO_DOCS)

    if args.verbose:
        print("Tokenized Doc:")
        print(doc_tokenizer.tok.decode(ids[0]))
    
    expected_doc_ids = torch.tensor([
        101,     2,  1996,  3739,  1997,  4807, 13463,  4045,  9273,  2001,
        8053,  2590,  2000,  1996,  3112,  1997,  1996,  7128,  2622,  2004,
        4045, 24823,  2001,  1012,  1996,  2069,  6112,  5689,  2058,  1996,
        8052,  6344,  1997,  1996,  9593,  6950,  1998,  6145,  2003,  2054,
        2037,  3112,  5621,  3214,  1025,  5606,  1997,  5190,  1997,  7036,
        3268, 27885, 22779,  9250,  1012,   102,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,     0,     0,     0,     0,     0
    ])

    expected_doc_mask =  torch.tensor([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0 
    ])

    assert (ids[0] == expected_doc_ids.to(ids.device)).all(), "Doc tokenized ids not as expected for colbertv2.0 on single example"

    assert (mask[0] == expected_doc_mask.to(mask.device)).all(), "Doc tokenized mask is not as expected for colbertv2.0 on single example"
    
    print("All tests passed!")


def main(args: argparse.Namespace) -> None:
    test_query_basic_tensorize(args)
    test_doc_basic_tensorize(args)
    test_tensorize_colbert_v2()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model checkpoint",
        default="colbert-ir/colbertv2.0"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="Whether to print the tokenized queries and documents",
    )
    args = parser.parse_args()
    main(args)
