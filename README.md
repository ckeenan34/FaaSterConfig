# FaaSterConfig
Automatic resource configuration generator for FaaS functions. Goal is to provide a way for a developer to automatically find the best (lowest time & cheapest cost) configurations for their function with various inputs. 

![Architecture](docs/Architecture.png)

This repo contains jupyter notebooks, openFaas functions, and script(s) to generate configs and interact with OpenFaas to determine which config is the best for a given function 

## Setup 

Lots of methods here will make use of various libraries, its good to just install everything from the requirements.txt file (optionally create a pyenv first)
```bash
pip install -r requirements.txt
```
### Notebooks

Start testing things with existing or create new jupyter notebooks and store them in the notebooks directory

### OpenFaaS

If you want to test out some of the functions by themselves, install faas-cli and use it to upload/run the functions in the openFaas directory

on mac: 

```bash
brew install faas-cli
cd openFaas
faas-cli local-run greeter
```

in a new terminal, send a curl request locally: 

```bash
curl http://0.0.0.0:8080 -X POST  --data 'ECHO'
```
This should return Echo to the terminal 

### FaaSterConfig

This python script will generate a set of configurations, save them in a _gen.yml file, then call the `faas-cli local-run` command to test that function locally. Future plans involve running this in an openFaas cluster and expanding what nodeTypes are available

```bash
cd FaaSterConfig
python3 FaaSterConfig.py ../openFaas/matmul.yml -c 1 6 12 -nt NoGPU -d 5000
```

This on in particular will run the matmul function on 1 6 and 12 cpus (default memory configs, check `python3 FaaSterConfig.py --help` for more options), on only one node type (no gpu since its local) and will send the function an input of "5000"

The output will look something like this and will be saved to a file for future reference: 

```txt
#Command: ./FaaSterConfig.py ../openFaas/matmul2.yml -d 1000 -nt r5.large c5.large m5.large c7g.large -c .25 .5 1 -m 248 500 15000 -to 120 -con 1 -v
#Function: matmul2
#Function argument(s): 1000
#Experiment cost: $0.18050395647511114
#Total time 639.025649, tradeoff: 0.5, timeout: 120.0, concurrency: 1
#Recommendation: CPU :1.0, Mem: 500.0, NodeType: c5.large which had a final time of: 0.5155s and expected cost of $0.00000473, with a combined z of -0.911
#Failures: 11.11%
#CPU: [0.25, 0.5, 1.0]
#Mem: [248.0, 500.0, 15000.0]
#NodeType: ['r5.large', 'c5.large', 'm5.large', 'c7g.large']
CPU,Mem,NodeTypeStr,time,cost,costPerHour,startupTime,timeAndCost
1.0,500.0,c5.large,0.515482,4.732051067776151e-06,0.0330474853515625,4.502557,-0.9106897192503463
1.0,248.0,m5.large,0.544234,4.80988056640625e-06,0.03181640625,3.364532,-0.8881009007266754
1.0,500.0,c7g.large,0.604672,4.721447317708333e-06,0.02810980224609375,2.290338,-0.8446337811850613
1.0,248.0,c5.large,0.606817,5.019565400865343e-06,0.029779052734375,3.411319,-0.8382912525752213
1.0,248.0,c7g.large,0.624784,4.395999646809896e-06,0.0253297119140625,3.536267,-0.8348855953679314
1.0,500.0,m5.large,0.652331,6.0996770751953124e-06,0.033662109375,5.621798,-0.7872873126781005
1.0,248.0,r5.large,0.65366,7.365843353271485e-06,0.0405670166015625,3.346955,-0.7661270265612173
1.0,15000.0,r5.large,0.611134,1.892355691833496e-05,0.1114727783203125,18.065191,-0.613557288657927
0.5,500.0,c7g.large,0.958519,4.4765306141662596e-06,0.016812927246093748,4.510169,-0.5857993789488292
1.0,15000.0,m5.large,0.609235,2.3669335042317705e-05,0.13986328125,5.621864,-0.5393541275887319
0.5,248.0,c7g.large,1.137977,4.4358460147094725e-06,0.0140328369140625,4.504927,-0.4531973953320104
0.5,500.0,m5.large,1.122903,5.821038500976562e-06,0.018662109375,3.391827,-0.4423201652787693
0.5,248.0,c5.large,1.144976,5.2471078287760416e-06,0.016497802734375,3.386702,-0.4350749204704505
0.5,248.0,m5.large,1.188834,5.55330986328125e-06,0.01681640625,3.401658,-0.3976310774063009
0.5,248.0,r5.large,1.209598,7.0155059783935555e-06,0.020879516601562502,3.335654,-0.3589167281320361
1.0,15000.0,c5.large,0.630411,3.8719854222615555e-05,0.221112060546875,4.47633,-0.28383479412641766
0.5,15000.0,r5.large,1.083886,2.7634660604858396e-05,0.0917852783203125,3.390519,-0.12374033257063602
0.5,15000.0,m5.large,1.075209,3.7292812158203124e-05,0.12486328125,16.145672,0.02369757826359678
0.25,248.0,c7g.large,1.843037,4.292432873026531e-06,0.0083843994140625,3.402971,0.06803504564832041
0.25,248.0,c5.large,1.900143,5.202790908813476e-06,0.009857177734375,3.403034,0.12494157706272929
0.25,500.0,c7g.large,1.901915,5.8983084765116365e-06,0.011164489746093749,4.613961,0.13733880596295495
0.25,248.0,m5.large,2.174281,5.6268014160156245e-06,0.00931640625,4.505374,0.33524871221401287
0.25,500.0,m5.large,2.167,6.718969726562499e-06,0.011162109375,4.488125,0.3472436844417838
0.25,248.0,r5.large,2.22056,6.807106079101562e-06,0.0110357666015625,3.327125,0.3884170235765079
0.5,15000.0,c7g.large,1.181042,5.799521254130045e-05,0.1767784423828125,4.48698,0.4321258686019902
0.25,500.0,r5.large,2.288924,7.786798184204102e-06,0.01224700927734375,3.34962,0.4547874662455102
0.5,15000.0,c5.large,1.098011,6.338903225538466e-05,0.207830810546875,3.392288,0.4564123992572787
0.25,500.0,c5.large,2.284045,8.327634637620715e-06,0.0131256103515625,4.725353,0.45978173771098224
0.25,15000.0,r5.large,2.26652,5.158947576904296e-05,0.0819415283203125,4.483629,1.1360483560232646
0.25,15000.0,m5.large,2.220465,7.238918286132812e-05,0.11736328125,4.486202,1.4332479373321898
0.25,15000.0,c7g.large,2.224204,0.00010573001149454751,0.1711300048828125,3.428957,1.9672345849651518
0.25,15000.0,c5.large,2.270587,0.00012689439439731175,0.201190185546875,3.369237,2.338881019550393
0.5,500.0,r5.large,,,0.02209075927734375,,
1.0,500.0,r5.large,,,0.04177825927734375,3.347983,
0.5,500.0,c5.large,,,0.0197662353515625,,
1.0,15000.0,c7g.large,,,0.1880753173828125,17.837452,
```



