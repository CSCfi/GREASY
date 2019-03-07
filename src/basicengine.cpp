/*
 * This file is part of GREASY software package
 * Copyright (C) by the BSC-Support Team, see www.bsc.es
 *
 * GREASY is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * GREASY is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GREASY. If not, see <http://www.gnu.org/licenses/>.
 *
*/


#include "basicengine.h"
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

BasicEngine::BasicEngine ( const string& filename, int ntasks_per_worker) :
  AbstractSchedulerEngine(filename, ntasks_per_worker) {

  engineType="basic";
  remote = false;
  this->ntasks_per_worker = ntasks_per_worker;
}

void BasicEngine::init() {


#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX sysconf (_SC_HOST_NAME_MAX)
#endif

  char hostname[HOST_NAME_MAX];

  log->record(GreasyLog::devel, "BasicEngine::init", "Entering...");

  AbstractSchedulerEngine::init();

  gethostname(hostname, sizeof(hostname));
  masterHostname=hostname;

  // Setup the nodelist
  if (config->keyExists("NodeList")) {
    log->record(GreasyLog::devel, "BasicEngine::init", "Nodelist setup");
    vector<string> nodelist = split(config->getValue("NodeList"),',');
    if (nworkers>nodelist.size()){
      log->record(GreasyLog::error,  "Requested more workers than nodes in the nodelist. Check parameters Nworkers and NodeList");
      ready = false;
    } else {
      // Fill the workerNodes map
      for (int i=1;i<=nworkers; i++) {
        workerNodes[i]=nodelist[i-1];
        if (!isLocalNode(workerNodes[i])) remote = true;
      }
    }

    if (remote) {
      if (config->keyExists("BasicRemoteMethod")) {
        if (config->getValue("BasicRemoteMethod")=="srun") {
          log->record(GreasyLog::debug,  "Using srun to spawn remote tasks");
        } else if (config->getValue("BasicRemoteMethod")=="ssh") {
          log->record(GreasyLog::debug,  "Using ssh to spawn remote tasks");
        } else {
          log->record(GreasyLog::error,  "No Basic Spawner in the system. Please run greasy locally unsetting the NodeList Parameter");
          ready = false;
        }
      } else {
         log->record(GreasyLog::error,  "No Basic Remote method configured. Please run greasy locally unsetting the NodeList Parameter");
         ready = false;
      }
    }
  }

  this->gpu_nodes = false;
  std::string command = "nvidia-smi &> /dev/null";
  int ret =  system(command.c_str());
  if (ret == 0) {
    log->record(GreasyLog::info,  "BasicEngine::init Detected GPU enabled compute nodes");
    this->gpu_nodes = true;
  }
  else {
    log->record(GreasyLog::error,  "BasicEngine::init Detected CPU only compute nodes");
    this->gpu_nodes = false;
  }

  if(this->gpu_nodes) {
    std::string command = "if [ ! -z ${CRAY_CUDA_MPS+x} ]; then /opt/cray/pdsh/default/bin/pdsh -w $SLURM_NODELIST '/etc/init.d/cray-nvidia-cuda-mps start &> /dev/null' &> /dev/null; fi";
    log->record(GreasyLog::debug,  "BasicEngine::init Started cray-nvidia-cuda-mps on all compute nodes ");
    system(command.c_str());
    //
    // pdsh always returns 0!!!
    //
    // int ret =  system(command.c_str());
    // if (ret == 0) {
    //   log->record(GreasyLog::info,  "BasicEngine::init cray-nvidia-cuda-mps started successfuly on all compute nodes");
    // }
    // else {
    //   log->record(GreasyLog::error,  "BasicEngine::init Failed to start cray-nvidia-cuda-mps on compute nodes");
    //   ready = false;
    // }
  }

  log->record(GreasyLog::devel, "BasicEngine::init", "Exiting...");

}

void BasicEngine::run() {

  log->record(GreasyLog::devel, "BasicEngine::run", "Entering...");

  if (isReady()) runScheduler();

  log->record(GreasyLog::devel, "BasicEngine::run", "Exiting...");

}

void BasicEngine::allocate(GreasyTask* task) {

  int worker;

  log->record(GreasyLog::devel, "BasicEngine::allocate", "Entering...");

  log->record(GreasyLog::info,  "Allocating task " + toString(task->getTaskId()));

  worker = freeWorkers.front();
  freeWorkers.pop();
  taskAssignation[worker] = task->getTaskId();

  pid_t pid = fork();

  if (pid == 0) {
    // Child:
    // Disable signal handling in child processes
    // We only want to have the master in charge of the restarts and messages.
    signal(SIGTERM, SIG_DFL);
    signal(SIGINT, SIG_DFL);

    // We use system instead of exec because of greater compatibility with command to be executed
    exit(executeTask(task,worker));
  } else if (pid > 0) {
    // Parent:
    log->record(GreasyLog::debug,  "BasicEngine::allocate", "Task "
          + toString(task->getTaskId()) + " to worker " + toString(worker)
          + " on node " + getWorkerNode(worker));
    pidToWorker[pid] = worker;
    task->setTaskState(GreasyTask::running);
    workerTimers[worker].reset();
    workerTimers[worker].start();

  } else {
   //error
   log->record(GreasyLog::error,  "Could not execute a new process");
   task->setTaskState(GreasyTask::failed);
   task->setReturnCode(-1);
   freeWorkers.push(worker);
  }

  log->record(GreasyLog::devel, "BasicEngine::allocate", "Exiting...");

}

void BasicEngine::waitForAnyWorker() {

  int retcode = -1;
  int worker;
  pid_t pid;
  int status;
  GreasyTask* task = NULL;

  log->record(GreasyLog::devel, "BasicEngine::waitForAnyWorker", "Entering...");

  // Wait for any of the worker to finish
  log->record(GreasyLog::debug,  "Waiting for any task to complete...");
  pid = wait(&status);

  // Identify the worker that was in charge of the child
  worker = pidToWorker[pid];

  // Get the return code
  if (WIFEXITED(status)) retcode = WEXITSTATUS(status);

  // Push worker to the free workers queue again
  freeWorkers.push(worker);
  workerTimers[worker].stop();

  // Update task info
  task = taskMap[taskAssignation[worker]];
  task->setReturnCode(retcode);
  task->setElapsedTime(workerTimers[worker].secsElapsed());
  task->setHostname(getWorkerNode(worker));

  // Run task epilogue stuff
  taskEpilogue(task);

  log->record(GreasyLog::devel, "BasicEngine::waitForAnyWorker", "Exiting...");

}

int BasicEngine::executeTask(GreasyTask *task, int worker) {

  log->record(GreasyLog::debug, "BasicEngine::executeTask["+toString(worker) +"]", "Entering...");
  string command = "";
  string node = "";
  int ret;

  if (!remote) {
    node = masterHostname;
  } else {
    node = workerNodes[worker];
  }

  if (config->getValue("BasicRemoteMethod")=="srun") {

    // capturing the number GREASY workers in a given compute node
    int num_same_nodes = 0;
    for(int i = 0; i < workerNodes.size(); i++) {
      if(workerNodes[i].compare(node) == 0) {
        num_same_nodes++;
      }
    }

    if(task->hasWorkDir()) {
      command += "cd " + task->getWorkDir() + " && ";
    }

    command += "srun -O -s -n " + toString(ntasks_per_worker) + " -N 1 -w " + node + " --cpu_bind=none ";

    if (ntasks_per_worker == 1) {
      command += "--gres=craynetwork:0";
    } else {
      command += "--gres=craynetwork:1";
    }

    // if(this->gpu_nodes) {
    //   command += ",gpu ";
    // } else {
    //   command += " ";
    // }
    command += " ";

    if (const char* SLURM_MEM_PER_NODE = std::getenv("SLURM_MEM_PER_NODE")) {
      log->record(GreasyLog::debug, "Memory on node " + node + " is " + string(SLURM_MEM_PER_NODE) + " MB");

      int memory = 0;
      fromString(memory, string(SLURM_MEM_PER_NODE));

      memory /= num_same_nodes;

      log->record(GreasyLog::debug, "Requesting memory of " + toString(memory) + "MB per task");
      command += "--mem=" + toString(memory) + " ";
    }


    if (const char* SLURM_CPUS_PER_TASK = std::getenv("SLURM_CPUS_PER_TASK")) {
      log->record(GreasyLog::debug, "CPUs per tasks: " + string(SLURM_CPUS_PER_TASK));

      int cpus_per_tasks = 0;
      fromString(cpus_per_tasks, string(SLURM_CPUS_PER_TASK));

      command += "-c " + toString(cpus_per_tasks) + " ";
    }

  } else if (config->getValue("BasicRemoteMethod")=="ssh") {

    if (ntasks_per_worker != 1) {
      log->record(GreasyLog::warning, "BasicEngine::executeTask["+toString(worker) +"]", "Ignoring request of " + toString(ntasks_per_worker) + " tasks per worker.");
    }

    command = "ssh -q " + node + " " ;

  } else {
    // Should never be here
    log->record(GreasyLog::error, "Unknown error. Check BasicRemoteMethod parameter.");
  }

  command += task->getCommand();
  log->record(GreasyLog::info,  "BasicEngine::executeTask[" + toString(worker) +"]", "Task "
          + toString(task->getTaskId()) + " on node " + node + " command: " + command);
  ret =  system(command.c_str());
  ret = WEXITSTATUS(ret);
  log->record(GreasyLog::debug, "BasicEngine::executeTask["+toString(worker) +"]", "Task returned "+toString(ret));
  log->record(GreasyLog::debug, "BasicEngine::executeTask["+toString(worker) +"]", "Exiting...");
  return ret;

}

bool BasicEngine::isLocalNode(string node) {

  return ((node == masterHostname)||(node == "localhost"));

}

string BasicEngine::getWorkerNode(int worker) {

  if (remote) return workerNodes[worker];
  else return masterHostname;

}
