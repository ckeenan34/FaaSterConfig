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
import random
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor  # or ProcessPoolExecutor if you want to use multiple processes
from doepy import build
import subprocess
import re
from tabulate import tabulate
import datetime


global_config_space = {
    'CPU': [.5,1,2],
    'Mem': [248, 1024, 1024*2],
    'NodeType': ["NoGPU"]#["GPU1", "GPU2", "GPU3"]
}

# Helpers
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

def genConfigs(config_space=None):
    if not config_space:
        config_space = global_config_space
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
        name = f"{funcName}-cpu{row.CPU}-mem-{row.Mem}-{row.NodeTypeStr}".replace(".", "x").lower()
        config = funcConfig.copy()
        memUnit = "m" if kwargs.get("local", False) else "Mi"
        config['limits'] = {
            'cpu': row.CPU,
            "memory": f"{row.Mem}{memUnit}"
        }
        config['requests'] = config['limits'].copy()
        functions[name] = config
        funcNames.append(name)
    doe['funcName'] = funcNames
    
    stack['functions'] = functions
    return doe, stack

def imageExists(image):
    """Checks if an image exists according to docker"""
    cmd = f"docker manifest inspect {image} > /dev/null ; echo $?"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (out, err) = process.communicate()
    if err:
        print(err)
    return out.strip() == "0"

def imagesExist(stackPath):
    """Checks if all images exist in the stack.yml file"""
    stack = readStack(stackPath)
    return all(map(imageExists, set(v['image'] for k,v in stack['functions'].items())))

def up(stackYml, verbose=False, **kwargs):
    """Deploys all functions in a stack.yml file to an OpenFaas cluster"""

    subcmd = "deploy" if imagesExist(stackYml) else "up"
    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli {subcmd} -f {stackYml}"
    if verbose: 
        print(f"running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = None
    for line in process.stdout:
        if verbose:
            print(line)
        if match:=re.search("Deployed. (\d+) Accepted.", line):
            result = f"Deployed {match.groups()[0]}"
        if match:=re.search("failed to deploy", line):
            raise Exception(f"Failed: {line}")
    print(f"up completed for {stackYml}")
    
    return result

def remove(stackYml, verbose=False):
    """Deletes all functions in a stack.yml file to an OpenFaas cluster"""

    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli rm -f {stackYml}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        if verbose:
            print(line)

    return True

def runLocalFunction(funcName=None, data="100", stackYml=None, port=4500, MAXBUF=120, verbose=False, **kwargs):
    assert funcName is not None, "funcName required"

    if not stackYml: 
        # Assume stack.yml file is same name as function
        stackYml = f"../openFaas/{funcName}.yml"

    logFile = f"/tmp/openFaas/{funcName}.log"
    os.makedirs(os.path.dirname(logFile), exist_ok=True)

    cmd = f"cd {os.path.dirname(stackYml)}; faas-cli local-run {funcName} -f {stackYml} -p {port} --quiet 2>&1 | tee {logFile}"
    if verbose:
        print("running command: ")
        print(cmd)
    process = subprocess.Popen(cmd, bufsize=MAXBUF, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    duration = None
    for line in process.stdout:
        if line:
            if verbose:
                print(line)
            if re.search("Listening on port: 8080", line):
                # TODO use requests and send payload from input 
                command = f"curl http://0.0.0.0:{port} -X POST  --data '{data}' >/dev/null 2>&1"
                if verbose:
                    print(f"Running command: {command}")
                os.system(command)
            elif (match:= re.search("Duration: ((?:\d|.)+)s", line)):
                duration = float(match.groups()[0])
                break

    if not process.poll():
        process.kill()
    # Ensure all subprocesses are killed for this function
    os.system("""kill -9 $(ps -ef | grep -e "faas-cli local-run %s" -e "docker run --name %s" -e "run.*%s" | awk '{print $2}')""" % (funcName,funcName,funcName))
    print(f"duration of {funcName}: {duration}s")
    return duration

def runFunction(funcName, baseUrl, data, verbose=False, **kwargs):
    # TODO get time from faas-cli logs instead of response.elapsed
    if verbose:
        print(datetime.datetime.now())
        print(f"using data: {data}")

    url = f"{baseUrl}/function/{funcName}"
    response = requests.post(url, data=data, timeout=kwargs.get('timeout', 60))
    
    if not response.ok:
        print(response.content)
        return None
    elif verbose:
        print(response.content)
    
    return response.elapsed.total_seconds()

def getTimes(doe, **kwargs):
    print(f"Getting times use data: {kwargs.get('data')}")

    # delete keys to avoid duplicate args issue
    kwargs = delKeys(kwargs, ['funcName'])

    def getLocalTime(row):
        return runLocalFunction(row.funcName, port=(4500+row.i), **kwargs)

    def getRemoteTime(row):
        return runFunction(row.funcName, kwargs['url'], **kwargs)
    
    getTime = getLocalTime if kwargs.get('local', False) else getRemoteTime

    results = executeThreaded(getTime, doe.itertuples(index=False), max_workers=12)
    doe['time'] = results
    return doe

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stackPath", type=str,
                        help="The .yml file with the initial OpenFaaS function definition")
    parser.add_argument("-f", "--funcName", type=str,
                        help="The function within the .yml file to optimize. Defaults to first function in stack.yml")
    parser.add_argument("-o", "--outStack", type=str,
                        help="The output .yml path where the configurations will be stored. Defaults to <stack>_gen.yml")
    parser.add_argument("-c", "--cpu", type=float, nargs="+",
                        help=f"The range of cpu values to test for. Defaults to {global_config_space['CPU']}")
    parser.add_argument("-m", "--mem", type=float, nargs="+",
                        help=f"The range of memory values to test for.Defaults to {global_config_space['Mem']}")
    parser.add_argument("-nt", "--nodeType", type=str, nargs="+",
                        help=f"The list of NodeTypes to test for. Defaults to {global_config_space['NodeType']}")
    parser.add_argument("-d", "--data", type=str, default="1000",
                        help=f"The input data to pass to the function to get the estimated runtime")
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction,
                        help=f"Verbose output, helpful for debugging. Defaults to False")
    parser.add_argument("-lc", "--local", action=argparse.BooleanOptionalAction,
                        help=f"Will only run the function locally using faas-cli local-run")
    parser.add_argument("-to", "--timeout", action=argparse.BooleanOptionalAction,
                        help=f"How long a function request will wait until existing and discounting it")
    parser.add_argument("-dry", "--dry-run", action=argparse.BooleanOptionalAction,
                    help=f"Dry run, will generate the stack_gen.yml file but will not run any function")

    args = parser.parse_args()

    config_space = {
        'CPU': args.cpu or global_config_space.get("CPU"),
        'Mem': args.mem or global_config_space.get("Mem"),
        'NodeType': args.nodeType or global_config_space.get("NodeType")
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

def localMain(args):
    # Generate a new stack file with new configs
    doe, stack = generateFunctionConfigs(**args)
    writeStack(stack, args['genStackPath'])
    print(f"Generated {len(doe)} configurations, testing on local now")

    doe = getTimes(doe, stackYml=args['genStackPath'], **args)
    doe.sort_values(by=["time"], inplace=True)
    print(tabulate(doe[['CPU','Mem','NodeTypeStr', 'time']].reset_index(drop=True), headers='keys', tablefmt='psql'))

    best = doe.iloc[0]
    print(f"Top Recommendation config: CPU :{best.CPU}, Mem: {best.Mem}, NodeType: {best.NodeTypeStr} which had a final time of: {best.time}s")
    
    # Clears up any docker instances
    os.system("docker stop $(docker ps | grep fwatchdog | awk '{print $1}') > /dev/null 2>&1")
    return args, doe, stack

def remoteMain(args):
    doe, stack = generateFunctionConfigs(**args)
    writeStack(stack, args['genStackPath'])
    print(f"Generated {len(doe)} configurations, testing on remote now")
    if not args.get("dry_run"):
        up(args['genStackPath'], **args)
        
        try:
            doe = getTimes(doe, **args)
        finally:
            print("not removing functions for now")
            # remove(args['genStackPath'], **args)
    else:
        doe["time"] = None
    doe.sort_values(by=["time"], inplace=True)
    print(tabulate(doe[['CPU','Mem','NodeTypeStr', 'time']].reset_index(drop=True), headers='keys', tablefmt='psql'))

    best = doe.iloc[0]
    print(f"Top Recommendation config: CPU :{best.CPU}, Mem: {best.Mem}, NodeType: {best.NodeTypeStr} which had a final time of: {best.time}s")
    return args, doe, stack

def main():
    # Get arguments
    args = parseArgs()
    main = localMain if args.get('local', False) else remoteMain
    main(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
