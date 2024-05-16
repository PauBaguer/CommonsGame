PWD=$(pwd)

docker run --gpus all -it --rm --privileged -v $PWD:/tmp/code PauBaguer/mypytorch bash /tmp/code/exec.sh