ngroups=("ng-9ab3e3a1", "ng-c5" "ng-c7" "ng-r5")

for ng in "${ngroups[@]}";
do
  echo "aws eks update-nodegroup-config --cluster-name ccc-cluster2 --nodegroup-name $ng --scaling-config minSize=$1,maxSize=4,desiredSize=$1"
  aws eks update-nodegroup-config --cluster-name ccc-cluster2 --nodegroup-name $ng --scaling-config minSize=$1,maxSize=4,desiredSize=$1
done