# Experiments we'll run with 

# XL
19 15 => 1140
-c 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 -m 256.0 512.0 768.0 1024.0 1280.0 1536.0 1792.0 2048.0 2304.0 2560.0 2816.0 3072.0 3328.0 3584.0 3840.0

# L 
10 8 => 320
-c 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 -m 256.0 768.0 1280.0 1792.0 2304.0 2816.0 3328.0 3840.0

# M 
5 4 => 80
-c 0.1 0.5 0.9 1.3 1.7 -m 256.0 1280.0 2304.0 3328.0

# S 
3 2 => 24
-c 0.1 0.9 1.7 -m 256.0 2304.0

# Matmul 
original large run: 
FaaSterConfig/results/FaaSterResults_matmul2_20231119-160223.csv




# Image_processing
./FaaSterConfig.py ../openFaas/image_processing.yml -d  https://raw.githubusercontent.com/MBtech/rethinking-serverless/main/benchmarks/face-detection/pigo-openfaas/samples/nasa.jpg -nt r5.large  -c 2 -m 2048 -to 60 -con 10

S: FaaSterResults_image-processing_20231119-200655.csv
M: FaaSterResults_image-processing_20231119-184711.csv

# Ocr 
./FaaSterConfig.py ../openFaas/ocr.yml -d https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg -nt m5.large  -c 1.5 -m 3000 -to 60 -con 10 -v

# s3 
./FaaSterConfig.py ../openFaas/s3.yml -d '{"input_bucket": "inputbucketbenchmark","object_key": "amzn_fine_food_reviews/reviews100mb.csv","output_bucket":"outputbucketbenchmark"}' -nt r5.large  -c 1 -m 2048 -to 60 -con 10
