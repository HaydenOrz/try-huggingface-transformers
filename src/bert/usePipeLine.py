from transformers import (
    pipeline
)

# maskedText = "Google is a [MASK] company!"
maskedText = "Google is headquartered in [MASK]."

unmasker = pipeline('fill-mask', model="google-bert/bert-base-cased")
list = unmasker(maskedText)

for item in list: 
    print(item)


