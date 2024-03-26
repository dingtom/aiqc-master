trtexec --explicitBatch --onnx=yolox_m_fabric.onnx \
        --minShapes=images:1x3x608x960 \
        --optShapes=images:16x3x608x960 \
        --maxShapes=images:32x3x608x960 \
        --fp16 \
        --saveEngine=yolox_m_fabric.engine \
        --shapes=images:16x3x608x960 \
        --device=0
