#!/bin/sh
"""":
if type python3 > /dev/null 2>&1
then
    exec python3 "$0" "$@"
elif type python > /dev/null 2>&1
then
    exec python "$0" "$@"
else
    echo >&2 "Python not installed"
fi
exit 1
""" #"

import os
import numpy as np
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor  # or ProcessPoolExecutor if you want to use multiple processes
from doepy import build
import subprocess
import re
from tabulate import tabulate
from datetime import datetime, timezone, timedelta
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval, STATUS_FAIL
from functools import partial
import pandas as pd
import sys

# region Helpers
global_config_space = {
    'CPU': [.5,1,2],
    'Mem': [248, 1024, 1024*2],
    'NodeType': [
        "m5.large",
        "c5.large",
        "r5.large",
        "c7g.large"
    ],
    'nodePrepend': "node.kubernetes.io/instance-type=", # The prepended string for nodeTypes
}

class Range(object):
    '''Helper class for argparse range of values https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-in-a-range-using-arg
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

def executeThreaded(func, args, max_workers=4):
    """Executes a function over a list of arguments in a threaded way 

    Args:
        func (Callable): The function that will be called on each args elem to produce a new value
        args: An array of arguments to pass to func
        max_workers (int): Number of threads to run at once

    Returns:
        df (pandas.DataFrame): The original dataframe with the new column
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, args))

    return results

def readStack(path='../openFaas/stack.yml'):
    """
    Read and parse a YAML file containing an OpenFaaS stack definition.

    This function opens and reads a YAML file located at the specified 'path' and parses its contents
    using the PyYAML library. The resulting data is typically an OpenFaaS stack definition.

    Parameters:
    path (str, optional): The path to the YAML file to be read (default is '../openFaas/stack.yml').

    Returns:
    dict: A Python dictionary representing the parsed contents of the YAML file.
    """
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        
    return data
    
def writeStack(data, path='../openFaas/stack.yml'):
    """
    Write a Python dictionary as a YAML file representing an OpenFaaS stack definition.

    This function takes a Python dictionary 'data' and writes it to a YAML file at the specified 'path'
    using the PyYAML library. The dictionary is typically used to represent an OpenFaaS stack definition.

    Parameters:
    data (dict): The Python dictionary representing the OpenFaaS stack definition to be written.
    path (str, optional): The path to the output YAML file (default is '../openFaas/stack.yml').
    """
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def delKeys(dict, keys):
    dict = dict.copy()
    for k in keys:
        if k in dict:
            del dict[k]
    return dict

def waitForResult(func, expected_result, timeout=120, poll_interval=1, *args, **kwargs):
    start_time = time.time()

    while True:
        result = func(*args, **kwargs)
        if result == expected_result:
            return result

        # print(f"waiting result: {result} doesn't match expected: {expected_result}")
        elapsed_time = time.time() - start_time

        if timeout is not None and elapsed_time >= timeout:
            raise TimeoutError(f"Function did not return the expected result {expected_result} within {timeout} seconds. Last result: {result}")

        time.sleep(poll_interval)

def runShell(cmd, handleLine=None, verbose = False):
    if verbose:
        print(cmd)
        #time.sleep(100)
        #print('time sleep over')
    process = subprocess.Popen(cmd, bufsize=120, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not handleLine:
        if verbose:
            print('in runShell no handleLine')
        (out, err) = process.communicate()
        if err:
            print(err)
        return out.strip()
    else: 
        if verbose:
            print('in runShell with handleLine')
        try:
            #if verbose:
            #    for line in process.stdout:
            #        print('runShell stdout: ', line)
            #    for line in process.stderr:
            #        print('runShell stderr: ', line)
            for line in process.stdout:
                #if verbose:
                #    print(line)
                if line and (r:=handleLine(line)):
                    return r
        finally:
            if not process.poll():
                process.kill()

def imageExists(image):
    """Checks if an image exists according to docker"""
    out = runShell(f"docker manifest inspect {image} > /dev/null ; echo $?")
    return out.strip() == "0"

def imagesExist(stackPath):
    """Checks if all images exist in the stack.yml file"""
    stack = readStack(stackPath)
    return all(map(imageExists, set(v['image'] for k,v in stack['functions'].items())))

def up(stackYml, funcs, verbose=False, waitUntilReady=False, **kwargs):
    """Deploys functions in a stack.yml file to an OpenFaas cluster"""

    subcmd = "deploy" if imagesExist(stackYml) else "up"
    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli {subcmd} -f {stackYml}"
    if funcs:
        cmd += f" --regex \"({'|'.join(funcs)})\""
    if verbose: 
        print(f"running command: {cmd}")

    def handleLine(line):
        if verbose:
            print(line)
        if match:=re.search("Deployed. (\d+) Accepted.", line):
            return f"Deployed {match.groups()[0]}"
        if match:=re.search("failed to deploy", line):
            print(f"Failed: {line}")
            return None
        
    result = runShell(cmd, handleLine)
    # print(f"result before waiting: {result}")
    if waitUntilReady:
        start = datetime.now(timezone.utc)
        # wait until each function is up
        for funcName in funcs:
            def functionStatus():
                status =  runShell(f"faas-cli describe -f {kwargs['genStackPath']} {funcName} | awk '/Status:/ {{print $2}}'").strip()
                #print('status in waitUntilReady is: ', status)
                return status
            try:
                # print(f"waiting for {funcName} to be Ready")
                waitForResult(functionStatus, "Ready", kwargs.get('timeout', 120))
            except TimeoutError as ex:
                return None
        result = (datetime.now(timezone.utc) - start).total_seconds()

    print(f"up completed for {stackYml} {' '.join(funcs) if funcs else ''} after : {result} sec")

    return result

def remove(stackYml, funcs, verbose=False, **kwargs):
    """Deletes functions in a stack.yml file to an OpenFaas cluster"""

    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli rm -f {stackYml}"
    if funcs:
      cmd += f" --regex \"({'|'.join(funcs)})\""
    if verbose:
        print("Running: " + cmd)
    
    if (out:=runShell(cmd)) and verbose:
        print(out)

    if funcs is not None:
        print('funcs type',type(funcs))
        print('funcs',funcs)
        
        funcName = funcs[0]
        print('!!! funcName is', funcName)
        status = "Not"
        while True:
            cmd = f"faas-cli describe -f {kwargs['genStackPath']} {funcName} | awk '/Status:/ {{print $2}}'"
            process = subprocess.Popen(cmd, bufsize=120, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
            (out, err) = process.communicate()
            try:
                status = out.strip()
                print(status)
                print(err)
            finally:
                if not process.poll():
                    process.kill()
            if status != 'Not':
                break
            time.sleep(1)


def getPerHourCosts():
    targetAverageUtilization = .8 # Assume to be 80% to simulate a real autoscaler 
    nodePerHour = {
        "m5.large": {
            'cost': 0.096,
            'cpu': 2,
            'mem': 8,
        },
        "c5.large": {
            'cost': 0.085,
            'cpu': 2,
            'mem': 4,
        },
        "r5.large": {
            'cost': 0.126,
            'cpu': 2,
            'mem': 16,
        },
        "c7g.large": {
            'cost': 0.0723,
            'cpu': 2,
            'mem': 4,
        }
    }
    cpuPerHour = {
        "fargate": 0.04048
    }
    memPerHour = {
        "fargate": 0.004445
    }
    for nt, p in nodePerHour.items():
        # Cost per cpu/mem is normalized by number of resources times target utilization rate
        cpuPerHour[nt] = p.get("cost") * .5/(p.get("cpu") * targetAverageUtilization)
        memPerHour[nt] = p.get("cost") * .5/(p.get("mem") * targetAverageUtilization)
    return cpuPerHour, memPerHour, nodePerHour

def getCost(doe, **kwargs):
    """
    Gets the cost given the runtime of the function using either fargate or a specific node type
    https://aws.amazon.com/ec2/pricing/on-demand/ https://aws.amazon.com/fargate/pricing/ 
    If its a node, cost is estimated based on the cpu and mem limits relative to overall mem and cpu of the node
    """
    cpuPerHour, memPerHour, _ = getPerHourCosts()
    def configCostPerHour(row):
        nodeType = row.NodeTypeStr if row.NodeTypeStr in cpuPerHour else 'fargate' # Assume fargate if no node s
        return row.CPU*cpuPerHour.get(nodeType) + (row.Mem * memPerHour.get(nodeType)/1024)
    
    def funcCost(row):
        if row.time is not None:
            return (row.costPerHour*row.time)/3600
        return np.inf

    doe['costPerHour'] = doe.apply(configCostPerHour, axis=1)
    doe['cost'] = doe.apply(funcCost, axis=1)
    return doe
# endregion 

def genConfigs(config_space=None):
    if not config_space:
        config_space = global_config_space
    config_space = config_space.copy()
    del config_space['nodePrepend'] # nodePrepend is not a real variable, should be ignored here

    config_space['CPU'] = np.array(config_space['CPU']).astype(np.float32)
    config_space['Mem'] = np.array(config_space['Mem']).astype(np.float32)
    doe = build.full_fact(config_space)

    doe['i'] = doe.index + 1
    doe['NodeTypeStr'] = list(map(lambda nt: config_space['NodeType'][int(nt)], doe['NodeType']))
    return doe

def generateFunctionConfigs(funcName, config_space=None, stackPath='../openFaas/stack.yml',**kwargs):    
    stack = readStack(stackPath)
    functions = stack['functions']
    funcConfig = functions[funcName]
    del functions[funcName]
    
    if not config_space:
        config_space = global_config_space
        
    doe = genConfigs(config_space)
    funcNames = []
    for _, row in doe.iterrows():
        cpuStr = f"{row.CPU:.3f}"
        memStr = f"{row.Mem:.3f}{'m' if kwargs.get('local', False) else 'Mi'}"

        name = f"{funcName}-cpu{cpuStr}-mem-{memStr}-{row.NodeTypeStr}".replace(".", "x").lower()
        config = funcConfig.copy()
        config['limits'] = {
            'cpu': cpuStr,
            "memory": memStr
        }
        config['requests'] = config['limits'].copy()
        config['limits']["constraints"] = [
            config_space.get('nodePrepend', '') + row.NodeTypeStr
        ]
        functions[name] = config
        funcNames.append(name)
    doe['funcName'] = funcNames
    
    stack['functions'] = functions
    return doe, stack

def runFunction(funcName, baseUrl, data, verbose=False, **kwargs):
    if verbose:
        print(datetime.now())
        print(f"using data: {data}")

    # deploy function 
    if not (st:= up(kwargs['genStackPath'], [funcName], waitUntilReady=True, verbose=verbose, **kwargs)):
        print(f"{funcName} failed to deploy")
        return None,None

    # Run function 
    start = datetime.now(timezone.utc)
    # overcome slight time difference in clocks
    logs_since_time = (start - timedelta(hours=0, minutes=1)).isoformat()
    url = f"{baseUrl}/function/{funcName}"
    print(f"About to request {url} with {data}")
    response = requests.post(url, data=data, timeout=kwargs.get('timeout', 120))
    
    if not response.ok:
        print(f"{funcName} Response code: {response.status_code} due to: {response.reason}, {response.content}")
        return None,None
    elif verbose:
        print('Response from openfaas running function: ', response.content)
    
    # get duration from logs
    def handleLine(line):
        if (match:= re.search("Duration: ((?:\d|.)+)s", line)):
            return float(match.groups()[0])
        
    duration = runShell(f"faas-cli logs {funcName} --since-time {logs_since_time} -g {baseUrl}", handleLine, True)

    remove(kwargs["genStackPath"], [funcName], verbose=verbose, **kwargs)
    print(f"Duration {duration} for {funcName}")
    return duration, st

def getTimeHyperoptObjective(x, doe, args, trials, space):
    print('in getTimesHyperoptObjective hyper params are: ', x)
    kwargs = args
 
    #x is: {'CPU': 2, 'Mem': 2048, 'NodeTypeStr': 'c5.large'}

    # skip duplicate run
    counter = 0
    for t in trials:
        if t['result']['status'] == STATUS_OK:
            vals = t["misc"]["vals"]
            rval = {}
            for k, v in list(vals.items()):
                if v:
                    rval[k] = v[0]
            if x == space_eval(space, rval):
                counter += 1
    if counter > 0:
        return {
            'status': STATUS_FAIL,
            }
    # delete keys to avoid duplicate args issue
    kwargs = delKeys(kwargs, ['funcName'])

    print(f"Getting times using input: {kwargs.get('data')}")

    def getLocalTime(row):
        # return runLocalFunction(row.funcName, port=(4500+row.i), **kwargs)
        raise "Local run is no longer supported"

    def getRemoteTime(row):
        return runFunction(row['funcName'].values[0], kwargs['url'], **kwargs)
    
    getTime = getLocalTime if kwargs.get('local', False) else getRemoteTime
    
    #get single row matching x for this iteration
    row = doe.loc[(doe['CPU'] == round(x['CPU'],1)) & (doe['Mem'] == x['Mem']) & (doe['NodeTypeStr'] == x['NodeTypeStr'])]
    row_i = row.index[0]

    time,startupTime = getTime(row)
    if kwargs.get("verbose", False):
        if not time and not startupTime:
            print("Results was empty")
        print("results time: ", time, "results startupTime: ", startupTime)

    #insert values at index
    doe.loc[row_i,'time'] = time
    doe.loc[row_i, 'startupTime'] = startupTime
    
    #return doe
    if kwargs.get("verbose", False): 
        print('in getTimesHyperoptObjective doe is: ', doe)

    # hyperopt will not integrate failed outcome
    if not time:
        print('in hyperopt return failed function')
        return {
            'status': STATUS_FAIL,
            }
    
    return {
        'loss': time,
        'status': STATUS_OK,
        # -- store other results like this
        #'doe': doe
        }
    

def getTimes(doe, **kwargs):
    print(f"Getting times using input: {kwargs.get('data')}")

    # delete keys to avoid duplicate args issue
    kwargs = delKeys(kwargs, ['funcName'])

    def getLocalTime(row):
        # return runLocalFunction(row.funcName, port=(4500+row.i), **kwargs)
        raise "Local run is no longer supported"

    def getRemoteTime(row):
        return runFunction(row.funcName, kwargs['url'], **kwargs)
    
    getTime = getLocalTime if kwargs.get('local', False) else getRemoteTime

    results = executeThreaded(getTime, doe.itertuples(index=False), max_workers=kwargs.get("concurrency", 30))
    if kwargs.get("verbose", False):
        if not results:
            raise "Results was empty"
        print(f"results length: {len(results)}")
    time,startupTime = zip(*results)
    doe['time'] = time
    doe['startupTime'] = startupTime
    return doe

def calcTimeAndCost(doe, tradeoff):
    def z(s):
        return (s - s.mean())/s.std()
    
    return (1-tradeoff)*z(doe['time']) + z(doe['cost'])*tradeoff

def calcExperimentCost(secElapsed, prefix='node.kubernetes.io/instance-type='):
    cmd = f"kubectl get nodes --show-labels | awk '{{print $6}}' | tr ',' '\n' | grep '{prefix}' | sort | uniq -c"

    nodeCounts = dict([(y[1].replace(prefix,''), y[0]) for y in [x.strip().split(" ") for x in runShell(cmd).split("\n")]])

    _,_,nodePerHour = getPerHourCosts()

    hourElapsed = (secElapsed+60)/3600

    cost = 0
    for node, count in nodeCounts.items():
        cost += int(count) * nodePerHour[node]['cost'] * hourElapsed

    # cost of eks during this time, assumes the cluster is only used for experiment 
    cost += .1 * hourElapsed
    return cost

def getHyperoptParams(config_space):
    #max_evals = round(hyperoptEvals * doe.shape[0])
    space = {'CPU' : hp.quniform('CPU_uniform', min(config_space['CPU']), max(config_space['CPU']), 0.1),
         'Mem': hp.choice('Mem_choice', config_space['Mem']),
         'NodeTypeStr': hp.choice('Node_choice', config_space['NodeType'])}
    return space

def remoteMain(args):
    doe, stack = generateFunctionConfigs(**args)
    writeStack(stack, args['genStackPath'])
    print(f"Generated {len(doe)} configurations, testing on remote now")
    start = datetime.now(timezone.utc)

    if args.get("dry_run"):
        print("skipping evaluation")
        return 
    try:
        if args["hyperoptEvals"] is not None:
            hyperoptEvals = min(args["hyperoptEvals"],doe.shape[0])

            space = getHyperoptParams(args['config_space'])

            trials = Trials()

            doe['time'] = pd.Series(dtype='float')
            doe['startupTime'] = pd.Series(dtype='float')

            print('max_evals are: ', hyperoptEvals)
            #data = {'doe': doe, 'args': args}
            #print('data is: ', data)
            
            fmin_objective = partial(getTimeHyperoptObjective, doe=doe, args=args, trials=trials, space=space)

            best = fmin(fmin_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=hyperoptEvals,
            trials=trials)

            #print('best is ', best)
            print('best eval is ', space_eval(space, best))
            #print(trials.results)

        else:
            doe = getTimes(doe, **args)
    finally:
        remove(args['genStackPath'], None, **args)

    experimentTime = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"Experiment took {experimentTime}s")

    doe = getCost(doe)
    doe['timeAndCost'] = calcTimeAndCost(doe, args.get("tradeoff"))

    #return results
    doe.sort_values(by=["timeAndCost"], inplace=True)

    res = doe[['CPU','Mem','NodeTypeStr', 'time', 'cost', 'costPerHour', 'startupTime', 'timeAndCost']].reset_index(drop=True)
    csvData = res.to_csv(index=False)
    if args.get("tablefmt", '') == 'csv':
        print(csvData)
    else:
        print(tabulate(res, headers='keys', tablefmt=args.get('tablefmt', 'psql'), showindex=False))

    best = doe.iloc[0]
    if not best.time:
        print("Something failed, best time was empty, exiting")
        return
    rec = f"CPU :{best.CPU}, Mem: {best.Mem}, NodeType: {best.NodeTypeStr} which had a final time of: {best.time:.4f}s and expected cost of ${best.cost:.8f}, with a combined z of {best.timeAndCost:.3f}"
    
    if args["hyperoptEvals"] is not None:
        skips = doe.shape[0] - hyperoptEvals
        failures = ((np.count_nonzero(res['time'].isnull()) - skips) / hyperoptEvals) * 100
    else:
        skips = 0
        failures = (res['time'].isnull().mean()) * 100

    failuresStr = f"Failures: {failures:.2f}%"
    skipStr = f"Skips: {skips}"
    print(failuresStr)
    print(skipStr)
    print(f"Top Recommendation config: {rec}")
    expCost = None
    try:  
        # expCost = (doe['costPerHour']/60 * doe['startupTime'].clip(lower=60) + doe['cost']).sum
        expCost = calcExperimentCost(experimentTime)
        print(f"This experiment costed an estimated total of: ${expCost}")
    except Exception as ex:
        print("Failed to compute experimental cost due to: {ex}")

    if not args.get("dry_run"):
        with open(f"results/FaaSterResults_{args.get('funcName')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",'w') as resFile:
            meta = [f'#{string}\n' for string in ([
                f"Command: {' '.join(sys.argv)}"
                f"Function: {args.get('funcName')}",
                f"Function argument(s): {args.get('data')}",
                f"Experiment cost: ${expCost}",
                f"Total time {experimentTime:5f}, tradeoff: ${args.get('tradeoff')}, timeout: ${args.get('timeout')}, concurrency: {args.get('concurrency')}",
                f"Recommendation: {rec}",
                failuresStr,
                skipStr
            ] + [
                f"{rtype}: {args.get('config_space', {}).get(rtype)}" for rtype in ["CPU", "Mem", "NodeType"]
            ])]
            resFile.writelines(meta + [csvData])
    return args, doe, stack, expCost

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stackPath", type=str,
                        help="The .yml file with the initial OpenFaaS function definition")
    parser.add_argument("-f", "--funcName", type=str,
                        help="The function within the .yml file to optimize. Defaults to first function in stack.yml")
    parser.add_argument("-os", "--outStack", type=str,
                        help="The output .yml path where the configurations will be stored. Defaults to <stack>_gen.yml")
    parser.add_argument("-c", "--cpu", type=float, nargs="+",
                        help=f"The range of cpu values to test for. Defaults to {global_config_space['CPU']}")
    parser.add_argument("-m", "--mem", type=float, nargs="+",
                        help=f"The range of memory values to test for.Defaults to {global_config_space['Mem']}")
    parser.add_argument("-nt", "--nodeType", type=str, nargs="+",
                        help=f"The list of NodeTypes to test for. Defaults to {global_config_space['NodeType']}")
    parser.add_argument("-np", "--nodePrepend", type=str, 
                        help=f"The string prepended to nodeType when writing to stack.yml. Defaults to {global_config_space['nodePrepend']}")
    parser.add_argument("-d", "--data", type=str, default="1000",
                        help=f"The input data to pass to the function to get the estimated runtime")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction,
                        help=f"Verbose output, helpful for debugging. Defaults to False")
    parser.add_argument("-lc", "--local", action=argparse.BooleanOptionalAction,
                        help=f"Will only run the function locally using faas-cli local-run")
    parser.add_argument("-to", "--timeout", type=float, default=60*5,
                        help=f"How long a function request will wait until exiting and discounting it (applies to deploy and run steps)")
    parser.add_argument("-dry", "--dry-run", action=argparse.BooleanOptionalAction,
                        help=f"Dry run, will generate the stack_gen.yml file but will not run any function")
    parser.add_argument("-tf", "--tablefmt", type=str, default="csv",
                        help=f"Formats the output table in any supported format by the tabulate function. Defaults to psql")
    parser.add_argument("-con", "--concurrency", type=int, default=30,
                        help=f"How many functions can be running at one time.")
    parser.add_argument("-trade", "--tradeoff", type=float, choices=[Range(0.0, 1.0)], default=.5,
                        help=f"Tradeoff between time (0) and cost (1)")
    parser.add_argument("-hy", "--hyperoptEvals", type=int,
                        help=f"number of trials to run of the total of doe, using hyperopt single threaded bayesian hyperparmeter optimization")

    args = parser.parse_args()

    config_space = {
        'CPU': args.cpu or global_config_space.get("CPU"),
        'Mem': args.mem or global_config_space.get("Mem"),
        'NodeType': args.nodeType or global_config_space.get("NodeType"),
        'nodePrepend': args.nodePrepend or global_config_space.get("nodePrepend"),
    }

    if not args.outStack:
        head, tail = os.path.splitext(args.stackPath)
        args.outStack = f"{head}_gen{tail}"
    elif os.path.isdir(args.outStack):
        head, tail = os.path.splitext(os.path.basename(args.stackPath))
        args.outStack = os.path.join(args.outStack, f"{head}_gen{tail}")

    stack = readStack(args.stackPath)
    if not stack.get('functions', False):
        return f"{args.stackPath} not valid"

    if not args.funcName:
        # Assume at least 1 function available in this stack
        args.funcName = next(iter(stack['functions'].items()))[0]

    finalArgs = vars(args) | {
        "genStackPath": args.outStack,
        "config_space": config_space,
        "url": stack.get("provider", {}).get("gateway", None)
    }
    if finalArgs['verbose']:
        print(f"finalArgs: {finalArgs}")
    return finalArgs

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    remoteMain(parseArgs())
