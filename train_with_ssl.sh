export NGPUS=4

######### VGG-16 SYNTHIA -> cityscapes
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_vgg16_src_synthia.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/vgg16_s2c_src/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_src_synthia.yaml resume results/vgg16_s2c_src/ OUTPUT_DIR results/vgg16_s2c_src/ SOLVER.BATCH_SIZE_VAL 1
python semantic_dist_init.py -cfg configs/deeplabv2_vgg16_src_synthia.yaml resume results/vgg16_s2c_src/model_best.pth OUTPUT_DIR results/vgg16_s2c_src/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_sdca.py -cfg configs/deeplabv2_vgg16_sdca_synthia.yaml resume results/vgg16_s2c_src/model_best.pth CV_DIR results/vgg16_s2c_src/ OUTPUT_DIR results/vgg16_s2c_ours/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_sdca_synthia.yaml resume results/vgg16_s2c_ours/ OUTPUT_DIR results/vgg16_s2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python pseudo_label.py -cfg configs/deeplabv2_vgg16_sdca_synthia.yaml DATASETS.TEST cityscapes_train resume results/vgg16_s2c_ours/model_best.pth OUTPUT_DIR results/vgg16_s2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/deeplabv2_vgg16_ssl_synthia.yaml PREPARE_DIR results/vgg16_s2c_ours/ OUTPUT_DIR results/vgg16_s2c_ours_ssl SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_ssl_synthia.yaml resume results/vgg16_s2c_ours_ssl/ OUTPUT_DIR results/vgg16_s2c_ours_ssl/ SOLVER.BATCH_SIZE_VAL 1


########## ResNet-101 SYNTHIA -> cityscapes
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_r101_src_synthia.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/r101_s2c_src/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_src_synthia.yaml resume results/r101_s2c_src/ OUTPUT_DIR results/r101_s2c_src/ SOLVER.BATCH_SIZE_VAL 1
python semantic_dist_init.py -cfg configs/deeplabv2_r101_src_synthia.yaml resume results/r101_s2c_src/model_best.pth OUTPUT_DIR results/r101_s2c_src/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_sdca.py -cfg configs/deeplabv2_r101_sdca_synthia.yaml resume results/r101_s2c_src/model_best.pth CV_DIR results/r101_s2c_src/ OUTPUT_DIR results/r101_s2c_ours/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_sdca_synthia.yaml resume results/r101_s2c_ours/ OUTPUT_DIR results/r101_s2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python pseudo_label.py -cfg configs/deeplabv2_r101_sdca_synthia.yaml DATASETS.TEST cityscapes_train resume results/r101_s2c_ours/model_best.pth OUTPUT_DIR results/r101_s2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/deeplabv2_r101_ssl_synthia.yaml PREPARE_DIR results/r101_s2c_ours/ OUTPUT_DIR results/r101_s2c_ours_ssl SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_ssl_synthia.yaml resume results/r101_s2c_ours_ssl/ OUTPUT_DIR results/r101_s2c_ours_ssl/ SOLVER.BATCH_SIZE_VAL 1


########## VGG-16 GTA5 -> cityscapes
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_vgg16_src.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/vgg16_g2c_src/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_src.yaml resume results/vgg16_g2c_src/ OUTPUT_DIR results/vgg16_g2c_src/ SOLVER.BATCH_SIZE_VAL 1
python semantic_dist_init.py -cfg configs/deeplabv2_vgg16_src.yaml resume results/vgg16_g2c_src/model_best.pth OUTPUT_DIR results/vgg16_g2c_src/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_sdca.py -cfg configs/deeplabv2_vgg16_sdca.yaml resume results/vgg16_g2c_src/model_best.pth CV_DIR results/vgg16_g2c_src/ OUTPUT_DIR results/vgg16_g2c_ours/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_sdca.yaml resume results/vgg16_g2c_ours/ OUTPUT_DIR results/vgg16_g2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python pseudo_label.py -cfg configs/deeplabv2_vgg16_sdca.yaml DATASETS.TEST cityscapes_train resume results/vgg16_g2c_ours/model_best.pth OUTPUT_DIR results/vgg16_g2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/deeplabv2_vgg16_ssl.yaml PREPARE_DIR results/vgg16_g2c_ours/ OUTPUT_DIR results/vgg16_g2c_ours_ssl SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_vgg16_ssl.yaml resume results/vgg16_g2c_ours_ssl/ OUTPUT_DIR results/vgg16_g2c_ours_ssl/ SOLVER.BATCH_SIZE_VAL 1


########### ResNet-101 GTA5 -> cityscapes
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_r101_src.yaml SOLVER.LAMBDA_LOV 0.75 OUTPUT_DIR results/r101_g2c_src SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_src.yaml resume results/r101_g2c_src/ OUTPUT_DIR results/r101_g2c_src/ SOLVER.BATCH_SIZE_VAL 1
python semantic_dist_init.py -cfg configs/deeplabv2_r101_src.yaml resume results/r101_g2c_src/model_best.pth OUTPUT_DIR results/r101_g2c_src/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_sdca.py -cfg configs/deeplabv2_r101_sdca.yaml resume results/r101_g2c_src/model_best.pth CV_DIR results/r101_g2c_src/ OUTPUT_DIR results/r101_g2c_ours/ SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_sdca.yaml resume results/r101_g2c_ours/ OUTPUT_DIR results/r101_g2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python pseudo_label.py -cfg configs/deeplabv2_r101_sdca.yaml DATASETS.TEST cityscapes_train resume results/r101_g2c_ours/model_best.pth OUTPUT_DIR results/r101_g2c_ours/ SOLVER.BATCH_SIZE_VAL 1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssl.py -cfg configs/deeplabv2_r101_ssl.yaml PREPARE_DIR results/r101_g2c_ours/ OUTPUT_DIR results/r101_g2c_ours_ssl SOLVER.BATCH_SIZE 8
python test.py -cfg configs/deeplabv2_r101_ssl.yaml resume results/r101_g2c_ours_ssl/ OUTPUT_DIR results/r101_g2c_ours_ssl/ SOLVER.BATCH_SIZE_VAL 1
