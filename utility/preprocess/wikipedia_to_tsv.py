import os
import ujson

from argparse import ArgumentParser
from colbert.utils.utils import print_message


def main(args):
    input_path = args.input
    output_path = args.output

    assert not os.path.exists(output_path), output_path

    RawCollection = []

    walk = [(dirpath, filenames) for dirpath, _, filenames in os.walk(input_path)]
    walk = sorted(walk)

    for dirpath, filenames in walk:
        print_message(f"#> Visiting {dirpath}")

        for filename in sorted(filenames):
            assert 'wiki_' in filename, (dirpath, filename)
            filename = os.path.join(dirpath, filename)

            print_message(f"#> Opening {filename} --- so far collected {len(RawCollection)} pages/passages")
            with open(filename) as f:
                for line in f:
                    RawCollection.append(ujson.loads(line))

    with open(output_path, 'w') as f:
        #line = '\t'.join(map(str, ['id', 'text', 'title'])) + '\n'
        #f.write(line)

        PID = 1

        for doc in RawCollection:
            title, text = doc['title'], doc['text']

            # Join sentences and clean text
            text = ' '.join(text.split())

            if args.keep_empty_pages or len(text) > 0:
                line = '\t'.join(map(str, [PID, text, title])) + '\n'
                f.write(line)

                PID += 1
    
    print_message("#> All done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="docs2passages.")

    # Input Arguments.
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--output', dest='output', required=True)

    parser.add_argument('--keep-empty-pages', dest='keep_empty_pages', default=False, action='store_true')

    args = parser.parse_args()

    main(args)
