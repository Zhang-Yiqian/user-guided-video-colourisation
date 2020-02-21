# user-guided-video-colourisation

CUDA_VISIBLE_DEVICES=0 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume encnet_jpu_res101_pcontext.pth.tar --split val --mode test --ms


CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume encnet_jpu_res101_pcontext.pth.tar --split val --mode test
