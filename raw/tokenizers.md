# Decode completion token IDs

The files under `raw/` contain generated token IDs and experiment metadata.
Generated text and duplicate prompt strings are omitted. Prompt inputs for fresh
model generation remain in `tasks/`.

Use the tokenizer corresponding to the model that produced the IDs. The
following cached tokenizer revisions were checked by decoding bundled samples:

| Model label in paths | Hugging Face tokenizer repository | Revision |
| --- | --- | --- |
| `qwen3_1p7b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `qwen3_4b` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `qwen3_8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `llama31_8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |

The C4 distinct-prompt dataset uses `Qwen/Qwen3-4B`. Both C4 bias-control
files use `meta-llama/Meta-Llama-3.1-8B-Instruct`.

## Example

Decoding needs only tokenizer files, not model weights or a GPU. Install the
optional dependencies separately from the figure-reproduction requirements:

```bash
python -m pip install tokenizers huggingface_hub
```

Run this example from the repository root. Change the raw-data path and the
repository/revision together to inspect another model.

```python
import gzip
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

path = "raw/c4/qwen3_4b/temp_1/base.jsonl.gz"
tokenizer_file = hf_hub_download(
    repo_id="Qwen/Qwen3-4B",
    filename="tokenizer.json",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
)
tokenizer = Tokenizer.from_file(tokenizer_file)

with gzip.open(path, "rt", encoding="utf-8") as stream:
    row = json.loads(next(stream))

# Base/instruction files nest IDs in completions; candidate-block and
# distinct-prompt files store a list of token-ID sequences directly.
if "completions" in row:
    token_ids = row["completions"][0]["token_ids"]
else:
    token_ids = row["token_ids"][0]

print(tokenizer.decode(token_ids, skip_special_tokens=False))
```

For an existing local tokenizer, use
`Tokenizer.from_file("/path/to/tokenizer.json")` instead of downloading it.
Downloading the Llama tokenizer requires access to its model repository.

Decoding reconstructs the text represented by the **retained IDs**, which may
be only a prefix of the original response. It cannot recover discarded tokens.
Special tokens such as `</think>` remain visible with `skip_special_tokens=False`;
set it to `True` for display without those tokens. Decoded output can differ from
historical text fields that omitted special tokens or contained longer responses.

Reproduction hashes the original IDs directly. Do not decode and re-encode them
as a preprocessing step: that can change the sequences and the results.
