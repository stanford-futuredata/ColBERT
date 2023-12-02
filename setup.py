import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='colbert-ir',
    version='0.2.14',
    author='Omar Khattab',
    author_email='okhattab@stanford.edu',
    description="Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stanford-futuredata/ColBERT',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "bitarray",
        "datasets",
        "flask",
        "git-python",
        "python-dotenv",
        "ninja",
        "scipy",
        "spacy",
        "tqdm",
        "transformers",
        "ujson"
    ], 
    extras_require={
        'faiss-gpu': ['faiss-gpu>=1.7.0'],
        'torch': ['torch==1.13.1']
    },
    include_package_data=True,
)
