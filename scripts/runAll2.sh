cd ../FaaSterConfig

# copying memory categories and cpu range from orig results
smHyperUniformCpuArgs="-c 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7" 
lxlHyperUniformCpuArgs="${smHyperUniformCpuArgs} 1.8 1.9"
lArgs="${lxlHyperUniformCpuArgs} -m 256.0 512.0 768.0 1024.0 1280.0 1536.0"
xlArgs="${lxlHyperUniformCpuArgs} -m 256.0 768.0 1280.0 1792.0 2304.0 2816.0 3328.0 3840.0"
mArgs="${smHyperUniformCpuArgs} -m 256.0 1280.0 2304.0 3328.0"
sArgs="${smHyperUniformCpuArgs} -m 256.0 2304.0"

extraArgs=" -to 120 -con 1"
matmulInputs=(10 1000 5000)

set -o xtrace
# for mi in "${matmulInputs[@]}";
# do
#   # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $sArgs $extraArgs #complete
#   # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $mArgs $extraArgs #complete
#   # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $lArgs $extraArgs
#   # ./FaaSterConfig.py ../openFaas/matmul2.yml -d $mi $xlArgs $extraArgs
# done

# re-running experiments from orig results 
# reducing the original runAll.sh number of trials or combinations.
# for small, reduce to 2/3 
# for medium, reduce to 1/2
# for xl, reduce to 1/3
./FaaSterConfig2.py ../openFaas/matmul2.yml -d 10 $sArgs $extraArgs -hy 16
./FaaSterConfig2.py ../openFaas/matmul2.yml -d 10 $mArgs $extraArgs -hy 40
./FaaSterConfig2.py ../openFaas/matmul2.yml -d 1000 $mArgs $extraArgs -hy 40
./FaaSterConfig2.py ../openFaas/matmul2.yml -d 5000 $sArgs $extraArgs -hy 16
./FaaSterConfig2.py ../openFaas/matmul2.yml -d 5000 $mArgs $extraArgs -hy 40
#./FaaSterConfig2.py ../openFaas/matmul2.yml -d 5000 $xlArgs $extraArgs -hy 106

./FaaSterConfig2.py ../openFaas/image_processing.yml -d https://raw.githubusercontent.com/MBtech/rethinking-serverless/main/benchmarks/face-detection/pigo-openfaas/samples/nasa.jpg $mArgs $extraArgs -hy 40
./FaaSterConfig2.py ../openFaas/image_processing.yml -d https://raw.githubusercontent.com/MBtech/rethinking-serverless/main/benchmarks/face-detection/pigo-openfaas/samples/nasa.jpg $sArgs $extraArgs -hy 16

./FaaSterConfig2.py ../openFaas/ocr.yml -d https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg $mArgs $extraArgs -hy 40
./FaaSterConfig2.py ../openFaas/s3.yml -d '{"input_bucket": "inputbucketbenchmark","object_key": "amzn_fine_food_reviews/reviews100mb.csv","output_bucket":"outputbucketbenchmark"}' $mArgs $extraArgs -hy 40
./FaaSterConfig2.py ../openFaas/ocr.yml -d https://www.pyimagesearch.com/wp-content/uploads/2017/06/tesseract_header.jpg $sArgs $extraArgs -hy 16
./FaaSterConfig2.py ../openFaas/s3.yml -d '{"input_bucket": "inputbucketbenchmark","object_key": "amzn_fine_food_reviews/reviews100mb.csv","output_bucket":"outputbucketbenchmark"}' $sArgs $extraArgs -hy 16

# testing
#./FaaSterConfig2.py ../openFaas/matmul2.yml -d 100 -c 0.9 1.0 -m 256.0 1280.0 $extraArgs -hy 4 --verbose #complete
#./FaaSterConfig2.py ../openFaas/matmul2.yml -d 100 $sArgs $extraArgs --verbose -hy 16 #complete

set +o xtrace

cd ../scripts

