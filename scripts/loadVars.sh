aws eks update-kubeconfig --region us-east-1 --name ccc-cluster

export OPENFAAS_URL=$(kubectl get svc -n openfaas gateway-external -o  jsonpath='{.status.loadBalancer.ingress[*].hostname}'):8080  \
&& echo Your gateway URL is: $OPENFAAS_URL

export PASSWORD=$(kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode) && \
echo "OpenFaaS admin password: $PASSWORD"

aws eks update-kubeconfig --region us-east-1 --name ccc-cluster2

export OPENFAAS_URL2=$(kubectl get svc -n openfaas gateway-external -o  jsonpath='{.status.loadBalancer.ingress[*].hostname}'):8080  && echo Your gateway URL is: $OPENFAAS_URL
export PASSWORD2=$(kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode) && \
echo "OpenFaaS admin password: $PASSWORD2"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 017884733126.dkr.ecr.us-east-1.amazonaws.com
