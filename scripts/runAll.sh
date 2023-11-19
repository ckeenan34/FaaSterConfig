cd ../FaaSterConfig

xlArgs="-c 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 -m 256.0 512.0 768.0 1024.0 1280.0 1536."
lArgs="-c 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 -m 256.0 768.0 1280.0 1792.0 2304.0 2816.0 3328.0 3840.0"
mArgs="-c 0.1 0.5 0.9 1.3 1.7 -m 256.0 1280.0 2304.0 3328.0"
sArgs="-c 0.1 0.9 1.7 -m 256.0 2304.0"

extraArgs="-to 240 -con 10 -dry"
matmulInputs=("10", "1000", "5000")

set -o xtrace
for mi in "${matmulInputs[@]}";
do
  ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $sArgs $extraArgs
  ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $mArgs $extraArgs
  # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $lArgs $extraArgs
  # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $xlArgs $extraArgs
done

./FaaSterConfig.py ../openFaas/image_processing.yml -d https://raw.githubusercontent.com/MBtech/rethinking-serverless/main/benchmarks/face-detection/pigo-openfaas/samples/nasa.jpg $mArgs $extraArgs
./FaaSterConfig.py ../openFaas/ocr.yml -d https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg $mArgs $extraArgs
./FaaSterConfig.py ../openFaas/s3.yml -d '{"input_bucket": "inputbucketbenchmark","object_key": "amzn_fine_food_reviews/reviews100mb.csv","output_bucket":"outputbucketbenchmark"}' $mArgs $extraArgs
./FaaSterConfig.py ../openFaas/image_processing.yml -d https://raw.githubusercontent.com/MBtech/rethinking-serverless/main/benchmarks/face-detection/pigo-openfaas/samples/nasa.jpg $sArgs $extraArgs
./FaaSterConfig.py ../openFaas/ocr.yml -d https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg $sArgs $extraArgs
./FaaSterConfig.py ../openFaas/s3.yml -d '{"input_bucket": "inputbucketbenchmark","object_key": "amzn_fine_food_reviews/reviews100mb.csv","output_bucket":"outputbucketbenchmark"}' $sArgs $extraArgs

set +o xtrace

cd ../scripts