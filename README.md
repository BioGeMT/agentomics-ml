# MultiAgent-ML

## Environment setup (Docker + conda environment)
Make sure you have Docker installed

```
docker build -t agents_img .
```


```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    agents_img
```

```
docker exec -it agents_cont bash
python tests/tests.py -v
source activate multiagent-ml-env
```

Create a `.env` file containing your API keys. Example content: `OPENAI_API_KEY=my-api-key-1234`

Run `python src/playground.py` to check everything works

## Guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`

