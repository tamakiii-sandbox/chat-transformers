# chat-transformers

- https://ollama.com/download/linux

## How to use
```sh
make install
time poetry run python main.py
```

## Setting HUGGING_FACE_HUB_TOKEN

```sh
export HUGGING_FACE_HUB_TOKEN=$(pbpaste)
echo $HUGGING_FACE_HUB_TOKEN | less
```
