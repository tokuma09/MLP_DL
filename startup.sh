#!bin/zsh

# dive into virtual env
source ./venv/bin/activate

# background running jupyter lab
nohup jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' >>/dev/null 2>&1 &

/bin/zsh
